import torch
import torch.nn as nn
from .models import TCN, save_model
from .utils import SpeechDataset, one_hot


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

    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th')))

#    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss = nn.CrossEntropyLoss()
    train_data = torch.utils.data.DataLoader(SpeechDataset('data/train.txt', transform=one_hot),
                                             batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_data = torch.utils.data.DataLoader(SpeechDataset('data/valid.txt', transform=one_hot),
                                             batch_size=args.batch_size, shuffle=False)

    global_step = 0
    for epoch in range(args.num_epoch):
        # Train for an epoch
        model.train()
        avg_lls = []
        for batch in train_data:
            batch = batch.to(device)
            logit = model(batch[:, :, :-1])
            loss_val = loss(logit, batch.argmax(dim=1))
            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            avg_lls.append(float(loss_val))

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        avg_train_ll = sum(avg_lls) / len(avg_lls)
        if train_logger is not None:
            train_logger.add_scalar('nll', avg_train_ll, global_step)

        # Evaluate
        model.eval()
        avg_lls = []
        for batch in valid_data:
            batch = batch.to(device)
            logit = model(batch[:, :, :-1])
            avg_lls.append(float(loss(logit, batch.argmax(dim=1))))
        avg_valid_ll = sum(avg_lls) / len(avg_lls)
        if valid_logger is not None:
            valid_logger.add_scalar('nll', avg_valid_ll, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t ll = %0.3f \t val ll = %0.3f' % (epoch, avg_train_ll, avg_valid_ll))

        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-1)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=128)

    args = parser.parse_args()
    train(args)
