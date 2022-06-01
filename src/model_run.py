import torch
from datetime import datetime
from util import AverageMeter

topk = (1, 5)

def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[k] = correct_k.mul_(100.0 / batch_size)
        return res


def model_run(
    data_loader,
    model,
    criterion,
    device,
    optimizer=None,
    print_every: int = 0
):

    if optimizer:
        model.train()
    else:
        model.eval()

    epoch_loss = AverageMeter('Loss', ':1.5f')
    epoch_acc_s = {k: AverageMeter(f'Acc@{k}', ':6.2f') for k in topk}

    for batch, (X, y_true) in enumerate(data_loader):
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        predict = model(X)
        loss = criterion(predict, y_true)

        batch_acc_s = accuracy(predict, y_true)
        for k in topk:
            epoch_acc_s[k].update(batch_acc_s[k].item(), y_true.size(0))

        if (print_every > 0) and ((batch+1) % print_every == 0):
            print(
                f'batch : {batch}, loss = {loss}, ')
            for k in topk:
                print(
                    f' Top-{k} accuracy = {batch_acc_s[k].item()} epoch : {epoch_acc_s[k]}')
        epoch_loss.update(loss.item(), X.size(0))

        if optimizer:
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return epoch_loss, epoch_acc_s


def train(
    train_loader,
    model,
    criterion,
    device,
    optimizer,
    print_every: int = 0,
):
    train_loss, train_acc = model_run(
        train_loader, model, criterion, device, optimizer, print_every)
    print(
        f'Train loss: {train_loss}\t'
        f'Train accuracy: {train_acc}\t')
    return train_loss, train_acc


def validate(
    valid_loader,
    model,
    criterion,
    device,
    print_every: int = 0,
):
    with torch.no_grad():
        valid_loss, valid_acc = model_run(valid_loader, model, criterion,
                                          device, print_every=print_every)
    print(
        f'Valid loss: {valid_loss}\t'
        f'Valid accuracy: {valid_acc}')

    return valid_loss, valid_acc
