import torch
import torch.nn as nn
from .models import TCN, save_model
from .utils import SpeechDataset, one_hot
import numpy as np

def train(args):
    from os import path
    import torch.utils.tensorboard as tb
    model = TCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your code from prior assignments
    Hint: SGD might need a fairly high learning rate to work well here

    """

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        torch.device('cpu')
    print('device = ', device)
    model.to(device)

    # Create the loss
    loss_function = torch.nn.CrossEntropyLoss().to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

    n_epochs = 20
    batch_size = 64

    data_train = list(SpeechDataset('data/train.txt'))
    data_valid = list(SpeechDataset('data/valid.txt'))

    global_step = 0
    for epoch in range(n_epochs):
        total_loss = []
        model.train()
        print("epoch: " + str(epoch))
        for i in range(0, len(data_train), batch_size):

            string_batch = data_train[i: i+batch_size]

            data = []
            label = []
            for item in string_batch:
                data.append(one_hot(item[:-1])[None])
                label.append(one_hot(item)[None])

            data = torch.cat(data, dim=0)
            label = torch.cat(label, dim=0)

            label = label.argmax(dim=1)

            data, label = data.to(device), label.to(device)

            pred = model.forward(data)

            loss = loss_function(pred, label)

            total_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Tensorboard logging
            train_logger.add_scalar("loss", loss, global_step=global_step)

            # Increase the global step
            global_step += 1

        # Calculate average epoch loss
        avg_train_loss = sum(total_loss) / len(total_loss)
        model.eval()

        total_loss_valid = []
        for j in range(0, len(data_valid), batch_size):
            string_batch = data_valid[j: j + batch_size]
            data = []
            label = []
            for item in string_batch:
                data.append(one_hot(item[:-1])[None])
                label.append(one_hot(item)[None])

            data = torch.cat(data, dim=0)
            label = torch.cat(label, dim=0)

            label = label.argmax(dim=1)

            data, label = data.to(device), label.to(device)

            pred = model.forward(data)

            loss_valid = loss_function(pred, label)

            total_loss_valid.append(loss_valid.item())

        avg_valid_loss = sum(total_loss_valid) / len(total_loss_valid)

        # Tensorboard logging
        train_logger.add_scalar("epoch_loss", avg_train_loss, global_step)
        valid_logger.add_scalar('valid_loss', avg_valid_loss, global_step)

        # Reduce learning rate if loss hasn't decreased for 10 epochs
        scheduler.step(avg_valid_loss)

        print('epoch %-3d \t train_loss = %0.3f \t valid_loss = %0.3f' % (epoch, avg_train_loss, avg_valid_loss))

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='logs')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
