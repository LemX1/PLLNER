import logging
import os
import sys
import time

import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils.CSIDN import extract_prior_knowledge, cal_transfer_matrix, beta_update, cal_Mu
from utils.evaluate import evaluate
from utils.LWLoss import confidence_update
from utils.load_model_and_parallel import load_model_and_parallel

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


class Trainer:
    def __init__(self,
                 model):
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.device = None

    def save(self, output_dir, step):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info(f'Saving checkpoint to {output_dir}')
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'step-{}.pt'.format(step)))
        logger.info('Checkpoint saved!')

    def build_optimizer_and_scheduler(self, args, total_steps):
        no_decay = ["bias", "LayerNorm.weight"]
        model_param = list(self.model.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []

        for name, para in model_param:
            space = name.split('.')
            if space[0] == 'bert_module':
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # bert other module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, 'lr': args.bert_lr},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': args.bert_lr},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, 'lr': args.other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': args.other_lr},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_lr, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps), num_training_steps=total_steps
        )

        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, train_dataset, dev_dataset=None, ent2id=None):
        loss_args = {
            'index': None
        }
        if not args.no_eval:
            assert dev_dataset is not None, 'dev_dataset should not be None when no_eval is False'
            assert ent2id is not None, 'ent2id should not be None when no_eval is False'
            best_step = 0
            best_p_r_f = []
            dev_criterion = -1
            criterion_dict = {
                'precision': 0,
                'recall': 1,
                'f1': 2
            }
            dev_loader = DataLoader(dataset=dev_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=0)

        self.model, self.device = load_model_and_parallel(self.model, args.gpu_ids)

        if args.use_leverage_weight or args.model == 'CSIDN_model':
            data_num = len(train_dataset)
            max_seq_len = 512
            entity_type_num = len(ent2id)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      sampler=SequentialSampler(train_dataset),
                                      num_workers=0)
            if args.use_leverage_weight:
                confidence = torch.ones(data_num, max_seq_len, entity_type_num)/entity_type_num
                confidence = confidence.to(self.device)
                loss_args.update({
                    'p_weight': args.p_weight,
                    'n_weight': args.n_weight,
                    'confidence': confidence,
                })
            if args.model == 'CSIDN_model':
                beta = torch.ones(data_num, max_seq_len, entity_type_num)
                beta = beta.to(self.device)
                logger.info('start calculating alpha matrix...')
                naive_classifier = load_model_and_parallel(args.naive_classifier_path, args.gpu_ids, model_type=args.naive_classifier_type, bert_dir=args.bert_dir, label_type_num=len(ent2id))[0]
                alpha_matrix, r_x, labels_num, naive_predictions = extract_prior_knowledge(naive_classifier, self.device, train_loader, len(ent2id))
                loss_args.update({
                    'transfer_matrix': torch.tensor([0])
                })
                logger.info('done calculating alpha matrix!')
        else:
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      sampler=RandomSampler(train_dataset),
                                      num_workers=0)
        total_steps = len(train_loader) * args.train_epochs

        self.build_optimizer_and_scheduler(args, total_steps)

        # Train
        logger.info("***** Running training *****")
        logger.info(f"  Num Examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.train_epochs}")
        logger.info(f"  Total training batch size = {args.batch_size}")
        logger.info(f"  Total optimization steps = {total_steps}")

        global_step = 0

        # print average loss every avg_loss_step
        avg_loss_step = 20
        avg_loss = 0.

        for epoch in range(args.train_epochs):
            logger.info('epoch:{}'.format(epoch))
            if args.model == 'CSIDN_model':
                transfer_matrix = cal_transfer_matrix(alpha_matrix, r_x, beta, labels_num)
                logger.info('transfer_matrix:{}'.format(transfer_matrix[8, 0:50, :]))
                loss_args['transfer_matrix'] = transfer_matrix
            for step, batch_data in enumerate(train_loader):

                self.model.zero_grad()
                self.model.train()
                loss_args['index'] = torch.arange(step * args.batch_size, step * args.batch_size + batch_data['token_ids'].shape[0])
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)

                if 0 <= args.n_weight_decay_step <= global_step:
                    loss_args['n_weight'] = loss_args['n_weight'] * pow(args.n_weight_decay_rate, (global_step / args.n_weight_decay_step))
                    logger.info('n_weight:{}'.format(loss_args['n_weight']))

                # print('step:{}, transfer_matrix:{}'.format(step, transfer_matrix[loss_args['index'], :]))
                loss = self.model(**loss_args, **batch_data)[0]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

                if args.use_leverage_weight or args.model == 'CSIDN_model':
                    with torch.no_grad():
                        emissions = self.model(mode='cal_emissions', **loss_args, **batch_data)[0]
                    if args.use_leverage_weight:
                        confidence = confidence_update(emissions, confidence=confidence, labels=batch_data['labels'], index=loss_args['index'])
                    if args.model == 'CSIDN_model':
                        beta[loss_args['index'], :] = beta_update(naive_predictions[loss_args['index'], :], emissions)
                global_step += 1

                if avg_loss_step == 1:
                    logger.info('Step: %d / %d ---> loss: %.5f' % (global_step, total_steps, loss.item()))
                elif global_step % avg_loss_step == 0:
                    avg_loss /= avg_loss_step
                    logger.info('Step: %d / %d ---> average loss: %.5f' % (global_step, total_steps, avg_loss))
                    avg_loss = 0.
                else:
                    avg_loss += loss.item()

                if global_step % args.eval_step == 0:
                    if args.no_eval:
                        continue
                    else:
                        p_r_f = evaluate(self.model, self.device, dev_loader, ent2id)
                        if p_r_f[criterion_dict[args.eval_criterion]] > dev_criterion:
                            dev_criterion = p_r_f[criterion_dict[args.eval_criterion]]
                            best_p_r_f = p_r_f
                            best_step = global_step
                        if not args.no_save:
                            self.save(args.output_dir, global_step)

        if not args.no_save and args.eval_step:
            if args.no_eval or args.eval_step > global_step:
                self.save(args.output_dir, global_step)
                best_step = global_step
            else:
                if not args.no_delete_not_optimal_model:
                    logger.info('deleting all not optimal model..')
                    model_list = os.listdir(args.output_dir)
                    for model in model_list:
                        if str(best_step) not in model:
                            os.system('rm -r {}/{}'.format(args.output_dir, model))
                    logger.info('all non-optimal model deleted!')
                logger.info('according to {}, model performs best in step {}, metrics:p={}, r={}, f1={}'.
                            format(args.eval_criterion, best_step, best_p_r_f[0], best_p_r_f[1], best_p_r_f[2]))

        torch.cuda.empty_cache()
        # torch.set_printoptions(profile='full')
        # print('\n\n')
        # print('-'*30 + 'confidence' + '-'*50)
        # print(confidence)
        # print('-'*30 + 'labels' + '-'*50)
        # print(train_dataset.labels)
        # print('\n\n')
        logger.info('Train done!')
        return best_step

