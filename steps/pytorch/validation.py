import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.autograd import Variable


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, output, target):
        prediction = self.sigmoid(output)
        return 1 - 2 * torch.sum(prediction * target) / (torch.sum(prediction) + torch.sum(target) + 1e-7)


def segmentation_loss(output, target, weight_bce=1.0, weight_dice=1.0):
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    return weight_bce*bce(output, target) + weight_dice*dice(output, target)


def cross_entropy(output, target, squeeze=False):
    if squeeze:
        target = target.squeeze(1)
    return F.nll_loss(output, target)


def mse(output, target, squeeze=False):
    if squeeze:
        target = target.squeeze(1)
    return F.mse_loss(output, target)


def multi_output_cross_entropy(outputs, targets):
    losses = []
    for output, target in zip(outputs, targets):
        loss = cross_entropy(output, target)
        losses.append(loss)
    return sum(losses) / len(losses)


def score_model(model, loss_function, datagen):
    batch_gen, steps = datagen
    partial_batch_losses = {}
    for batch_id, data in enumerate(batch_gen):
        X = data[0]
        targets_tensors = data[1:]

        if torch.cuda.is_available():
            X = Variable(X, volatile=True).cuda()
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor, volatile=True).cuda())
        else:
            X = Variable(X, volatile=True)
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor, volatile=True))

        outputs = model(X)
        if len(loss_function) == 1:
            for (name, loss_function_one, weight), target in zip(loss_function, targets_var):
                loss_sum = loss_function_one(outputs, target) * weight
        else:
            batch_losses = []
            for (name, loss_function_one, weight), output, target in zip(loss_function, outputs, targets_var):
                loss = loss_function_one(output, target) * weight
                batch_losses.append(loss)
                partial_batch_losses.setdefault(name, []).append(loss)
            loss_sum = sum(batch_losses)
        partial_batch_losses.setdefault('sum', []).append(loss_sum)
        if batch_id == steps:
            break
    average_losses = {name: sum(losses) / steps for name, losses in partial_batch_losses.items()}
    return average_losses


def torch_acc_score(output, target):
    output = output.data.cpu().numpy()
    y_true = target.numpy()
    y_pred = output.argmax(axis=1)
    return accuracy_score(y_true, y_pred)


def torch_acc_score_multi_output(outputs, targets, take_first=None):
    accuracies = []
    for i, (output, target) in enumerate(zip(outputs, targets)):
        if i == take_first:
            break
        accuracy = torch_acc_score(output, target)
        accuracies.append(accuracy)
    avg_accuracy = sum(accuracies) / len(accuracies)
    return avg_accuracy
