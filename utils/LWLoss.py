import sys

import torch
import torch.nn.functional as F


def confidence_update(outputs, confidence, labels, index):
    with torch.no_grad():
        device = outputs.device
        softmax_outputs = F.softmax(outputs, dim=-1)

        negative = torch.zeros(softmax_outputs.shape[0], softmax_outputs.shape[1], softmax_outputs.shape[2])
        negative[labels < 1] = 1
        negative = negative.to(device)

        labels[labels == 0] = -10000
        labels = torch.softmax(labels.float(), dim=-1)
        new_weight_p = softmax_outputs * labels
        new_weight_p = new_weight_p / (new_weight_p + 1e-8)\
            .sum(dim=-1).unsqueeze(-1).repeat(1, 1, confidence.shape[-1])
        new_weight_n = softmax_outputs * negative
        new_weight_n = new_weight_n / (new_weight_n + 1e-8)\
            .sum(dim=-1).unsqueeze(-1).repeat(1, 1, confidence.shape[-1])
        new_weight = new_weight_n + new_weight_p
        confidence[index, :] = new_weight

        return confidence


# def LWLoss_with_sigmoid(outputs, labels, confidence, step, p_weight, n_weight):
#     device = outputs.device
#     index = torch.arange(step * 8, step*8+outputs.shape[0])
#
#     negative = torch.zeros(outputs.shape[0], outputs.shape[1], outputs.shape[2])
#     negative[labels < 1] = 1
#     negative = negative.to(device)
#
#     p_loss = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1], outputs.shape[2])
#     p_loss = p_loss.to(device)
#     p_loss[outputs < 0] = 1 / (1 + torch.exp(outputs[outputs < 0]))
#     p_loss[outputs > 0] = torch.exp(-outputs[outputs > 0]) / (1 + torch.exp(-outputs[outputs > 0]))
#     l1 = confidence[index, :] * labels * p_loss
#     p_average_loss = torch.sum(l1) / (l1.size(0)*l1.size(1))
#
#     n_loss = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1], outputs.shape[2])
#     n_loss = n_loss.to(device)
#     n_loss[outputs > 0] = 1 / (1 + torch.exp(-outputs[outputs > 0]))
#     n_loss[outputs < 0] = torch.exp(outputs[outputs < 0] / (1 + torch.exp(outputs[outputs < 0])))
#     l2 = confidence[index, :] * negative * n_loss
#     n_average_loss = torch.sum(l2) / (l2.size(0)*l2.size(1))
#
#     average_loss = p_weight * p_average_loss + n_weight * n_average_loss
#     return average_loss


def LWLoss_with_cross(outputs, labels, confidence, index, p_weight, n_weight):
    device = outputs.device

    negative = torch.zeros(outputs.shape[0], outputs.shape[1], outputs.shape[2])
    negative[labels < 1] = 1
    negative = negative.to(device)

    sm_outputs = F.softmax(outputs, dim=-1)

    labels[labels == 0] = -10000
    labels = torch.softmax(labels.float(), dim=-1)
    p_loss = -torch.log(sm_outputs + 1e-8)
    l1 = confidence[index, :] * labels * p_loss
    p_average_loss = torch.sum(l1) / l1.size(0)

    n_loss = -torch.log(1 - sm_outputs + 1e-8)
    l2 = confidence[index, :] * negative * n_loss
    n_average_loss = torch.sum(l2) / l2.size(0)

    average_loss = p_weight * p_average_loss + n_weight * n_average_loss
    return average_loss
