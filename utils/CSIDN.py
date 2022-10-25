import sys

import torch
import torch.nn.functional as F


def extract_prior_knowledge(model, device, data_loader, entity_type_num, seq_length=512):
    # initial
    labels = torch.zeros(entity_type_num)
    labels = labels.to(device)
    labels_num = torch.zeros(entity_type_num)
    labels_num = labels_num.to(device)
    r_x = []
    average_emissions = torch.zeros(entity_type_num)
    average_emissions = average_emissions.to(device)
    naive_predictions = []

    model.eval()
    with torch.no_grad():
        for batch_data in data_loader:
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            emissions = model(mode='evaluate', **batch_data)[1]
            emissions = F.softmax(emissions, dim=-1)
            naive_predictions.append(emissions)

            # sum of r_x * h_noisy(x)_i
            index1 = []
            for i in range(batch_data['labels'].shape[0]):
                index1 += [i] * seq_length
            index1 = torch.tensor(index1)
            index2 = torch.arange(0, seq_length, 1).repeat(batch_data['labels'].shape[0])
            index3 = batch_data['labels'].view(-1)
            labels_batch = torch.zeros(emissions.size())
            labels_batch = labels_batch.to(device)
            labels_batch[index1, index2, index3] = emissions[index1, index2, index3]
            r_x_batch = torch.zeros(emissions.size())
            r_x_batch[labels_batch > 0] = 0.8
            r_x.append(r_x_batch)
            labels += (labels_batch * 0.8).sum(1).sum(0)

            # num of label i
            labels_num = labels_num + batch_data['labels'].view(-1).bincount(minlength=entity_type_num)

            # sum of h_noisy(x)_j
            emissions[labels_batch > 0] = 0
            average_emissions = average_emissions + emissions.sum(1).sum(0)

        labels = (1 - (labels / labels_num)).unsqueeze(1).repeat(1, entity_type_num)
        average_emissions = (average_emissions / labels_num).unsqueeze(0).repeat(entity_type_num, 1)
        # alpha_matrix = average_emissions / labels
        alpha_matrix = torch.ones(7, 7)
        alpha_matrix = alpha_matrix / 6
        alpha_matrix = alpha_matrix.to(device)
        r_x = torch.cat(r_x, dim=0)
        r_x = r_x.to(device)
        naive_predictions = torch.cat(naive_predictions, dim=0)
        return alpha_matrix, r_x, labels_num, naive_predictions


def cal_Mu(r_x, beta, labels_num):
    print('total probability:', (r_x * beta).sum(1).sum(0))
    print('Mu:', (r_x * beta).sum(1).sum(0) / labels_num)
    return (r_x * beta).sum(1).sum(0) / labels_num


def cal_transfer_matrix(alpha_matrix, r_x, beta, labels_num):

    # cal diagonal value of candidate label
    diagonal = r_x * beta

    # T[i, j] = Alpha[i, j] * (1 - T[i, i]), i != j
    non_diagonal = (1 - diagonal).unsqueeze(2).repeat(1, 1, labels_num.shape[-1], 1)
    # print('non_diagonal:', non_diagonal[8, :])
    # print('non_diagonal_size:', non_diagonal.size())
    # print('non_diagonal1:', non_diagonal[8, :])
    # print('mul2_size:', alpha_matrix.unsqueeze(0).unsqueeze(0).repeat(non_diagonal.shape[0], non_diagonal.shape[1], 1, 1).size())
    # print('before softmax:', (non_diagonal * alpha_matrix.unsqueeze(0).unsqueeze(0).repeat(non_diagonal.shape[0], non_diagonal.shape[1], 1, 1))[8, :])
    non_diagonal = F.softmax(non_diagonal * alpha_matrix.unsqueeze(0).unsqueeze(0).repeat(non_diagonal.shape[0], non_diagonal.shape[1], 1, 1), dim=-1)
    # print('alpha_matrix:', alpha_matrix.unsqueeze(0).unsqueeze(0).repeat(non_diagonal.shape[0], non_diagonal.shape[1], 1, 1)[8, :])
    # print('non_diagonal2:', non_diagonal[8, :])

    # cal diagonal value of false label
    diagonal[diagonal == 0] = cal_Mu(r_x, beta, labels_num).unsqueeze(0).unsqueeze(0).repeat(r_x.shape[0], r_x.shape[1], 1)[diagonal == 0]

    # combine diagonal part and non-diagonal part
    transfer_matrix = torch.diag_embed(diagonal)
    transfer_matrix[transfer_matrix == 0] = non_diagonal[transfer_matrix == 0]

    return transfer_matrix


def CSIDN_Loss(emissions, transfer_matrix, labels, seq_length=512):
    emissions = F.softmax(emissions, dim=-1)
    # print('emissions:', emissions)
    # print('e_size:', emissions.size())
    # print('l_size:', labels.size())
    # print('transfer_matrix:', transfer_matrix)
    index3 = labels.view(-1)
    index2 = torch.arange(0, seq_length, 1).repeat(labels.shape[0])
    index1 = []
    for i in range(labels.shape[0]):
        index1 += [i] * seq_length
    index1 = torch.tensor(index1)
    labels = torch.zeros(labels.shape[0], seq_length, emissions.shape[-1])
    labels = labels.to(emissions.device)
    labels[index1, index2, index3] = 1
    print('loss before sum:', labels * torch.matmul(emissions.unsqueeze(2), transfer_matrix).squeeze())
    print('size:', torch.matmul(emissions.unsqueeze(2), transfer_matrix).squeeze().size())
    loss = (labels * torch.matmul(emissions.unsqueeze(2), transfer_matrix).squeeze()).sum()
    return loss


def beta_update(naive_predictions, emissions):
    # a = (naive_predictions + 1e-8) / F.softmax(emissions, dim=-1)
    # a = F.softmax(emissions, dim=-1)
    # print('beta:', a)
    # print(a.size())
    return (naive_predictions + 1e-8) / F.softmax(emissions, dim=-1)
