import torch
import torch.nn.functional as F


def extract_prior_knowledge(model, device, data_loader, entity_type_num, seq_length=512):
    # initial
    labels = torch.zeros(entity_type_num)
    labels = labels.to(device)
    labels_num = torch.zeros(entity_type_num)
    labels_num = labels_num.to(device)
    r_x = []
    naive_predictions = []

    model.eval()
    with torch.no_grad():
        wrong_labels = torch.zeros(entity_type_num, entity_type_num)
        wrong_labels = wrong_labels.to(device)
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
            for i in range(emissions.shape[0]):
                for j in range(emissions.shape[1]):
                    subscripts = torch.nonzero(labels_batch[i][j], as_tuple=True)[0]
                    for subscript in subscripts:
                        wrong_labels[subscript] += emissions[i][j]
                        wrong_labels[subscript][subscript] = 0

        labels = (1 - (labels / labels_num)).unsqueeze(1).repeat(1, entity_type_num)
        wrong_labels = (wrong_labels / labels_num.unsqueeze(1).repeat(1, entity_type_num))
        alpha_matrix = (wrong_labels / labels).transpose(dim0=-1, dim1=-2)
        r_x = torch.cat(r_x, dim=0)
        r_x = r_x.to(device)
        naive_predictions = torch.cat(naive_predictions, dim=0)
        return alpha_matrix, r_x, labels_num, naive_predictions


def cal_Mu(r_x, beta, labels_num):
    mu = (r_x * beta).sum(1).sum(0) / labels_num
    return mu


def cal_transfer_matrix(alpha_matrix, r_x, beta, labels_num):

    # cal diagonal value of candidate label
    diagonal = r_x * beta

    # T[i, j] = Alpha[i, j] * (1 - T[i, i]), i != j
    non_diagonal = (1 - diagonal).unsqueeze(2).repeat(1, 1, labels_num.shape[-1], 1)
    non_diagonal = F.softmax(non_diagonal * alpha_matrix.unsqueeze(0).unsqueeze(0).repeat(non_diagonal.shape[0], non_diagonal.shape[1], 1, 1), dim=-1)

    # cal diagonal value of false label
    diagonal[diagonal == 0] = cal_Mu(r_x, beta, labels_num).unsqueeze(0).unsqueeze(0).repeat(r_x.shape[0], r_x.shape[1], 1)[diagonal == 0]

    # combine diagonal part and non-diagonal part
    transfer_matrix = torch.diag_embed(diagonal)
    transfer_matrix[transfer_matrix == 0] = non_diagonal[transfer_matrix == 0]
    transfer_matrix = F.softmax(transfer_matrix, dim=-1)

    return transfer_matrix


def CSIDN_Loss(emissions, transfer_matrix, labels, seq_length=512):
    emissions = F.softmax(emissions, dim=-1)
    index3 = labels.view(-1)
    index2 = torch.arange(0, seq_length, 1).repeat(labels.shape[0])
    index1 = []
    for i in range(labels.shape[0]):
        index1 += [i] * seq_length
    index1 = torch.tensor(index1)
    loss = torch.matmul(emissions.unsqueeze(2), transfer_matrix).squeeze()
    loss = -torch.log(loss[index1, index2, index3]).sum()

    return loss


def beta_update(naive_predictions, emissions):
    return (naive_predictions + 1e-8) / F.softmax(emissions, dim=-1)
