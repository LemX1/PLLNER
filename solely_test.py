from utils.load_model_and_parallel import load_model_and_parallel
import argparse
import json
import os
from utils.processor import raw_data_to_dataset
from torch.utils.data import DataLoader, RandomSampler
from utils.evaluate import evaluate
import logging
    
    
def test(model_path, model_type, gpu_ids, data_dir, ent2id, bert_dir, label_type_num, strict=True, ent2id_way=None, batch_size=64):
    if type(ent2id) == str:
        assert ent2id_way is not None
        ent2id = json.load(open(os.path.join(ent2id, ent2id_way + '_ent2id.json')))
    model, device = load_model_and_parallel(model_path, gpu_ids, model_type, strict, bert_dir, label_type_num)
    test_dataset = raw_data_to_dataset(data_dir, bert_dir, 'test.json', 'test', ent2id)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)
    evaluate(model, device, test_loader, ent2id)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='out', type=str,
                        help='Path of model needs to be tested')
    parser.add_argument('--gpu_ids', default='1', type=str,
                        help='gpu ids to use, -1 for cpu, "0,1,……" for multi gpu')
    parser.add_argument('--data_dir', default='data/processed', type=str)
    parser.add_argument('--bert_dir', default='pretrain/torch_roberta_wwm', type=str)
    parser.add_argument('--ent2id_way', default='bio', type=str, choices=['bio', 'bioes'])
    parser.add_argument('--ent2id', default='model', type=str)
    parser.add_argument('--label_type_num', default=7, type=str,
                        help='number of label, all entity type except O may have 3(bio) or 5(bioes) kinds of label')
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    
    logger.info('----------------start testing----------------')
    # test(args.model_path, args.gpu_ids, args.data_dir, args.ent2id_dir,
    #      args.bert_dir, args.label_type_num, ent2id_way=args.ent2id_way)
    test(**vars(args))
    logger.info('----------------test done----------------')
