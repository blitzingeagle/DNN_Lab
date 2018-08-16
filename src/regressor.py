from __future__ import print_function
import argparse
import cv2
import numpy as np
import os
import sys
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sig(x)
        return x


def cv2_to_pytorch(img):
    data = img[:, :, ::-1]
    data = np.swapaxes(np.swapaxes(np.array(data, dtype=float), 0, 2), 1, 2) / 255.
    data_shape = (1,) + data.shape
    data = torch.from_numpy(data.reshape(data_shape)).float()

    return data


def data_from_filepath(filepath, img_shape=(32, 32)):
    data = cv2.resize(cv2.imread(filepath), img_shape)
    data = cv2_to_pytorch(data)

    return data


def load_data(dataroot, img_shape=(32, 32), labeled=True):
    labels_filepath = os.path.join(dataroot, "labels.txt")

    loader = []

    with open(labels_filepath, "r") as f:
        contents = f.readlines()
        for line in contents:
            line = line.strip().split()

            if labeled:
                (filename, label) = (line[0], line[1])
                filepath = os.path.join(dataroot, filename)

                data = data_from_filepath(filepath, img_shape=img_shape)
                target = torch.tensor([[float(label)]])
                loader.append((data, target))
            else:
                filepath = os.path.join(dataroot, line[0])
                data = cv2.resize(cv2.imread(filepath), img_shape)
                data = cv2_to_pytorch(data)
                loader.append(data)

    return loader


def train(args, model, device, optimizer, criterion, train_loader):
    model.train()

    model_path = os.path.join("models", args.name)
    checkpoints_path = os.path.join(model_path, "checkpoints")
    log_filepath = os.path.join(model_path, "log.txt")
    weights_filepath = os.path.join(model_path, "weights.pth")

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    with open(log_filepath, "w") as log:
        for epoch in range(1, args.epochs + 1):
            for (batch_idx, (data, target)) in enumerate(train_loader, 1):
                (data, target) = (data.to(device), target.to(device))
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % args.log_interval == 0:
                    print("Train Epoch: {} [{:3d}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss.item()
                    ))
                    log.write("Train Epoch: {} [{:3d}/{} ({:.0f}%)]\tLoss: {:.6f}\n".format(
                        epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss.item()
                    ))

            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(checkpoints_path, "save_%06d.pth" % epoch))

        torch.save(model.state_dict(), weights_filepath)


def test(args, model, device, test_list):
    model.eval()

    results_path = os.path.join("results", args.name, os.path.basename(args.dataroot))
    label_filepath = os.path.join(results_path, "labels.txt")

    os.makedirs(results_path, exist_ok=True)

    with open(label_filepath, "w") as labels:
        for filepath in test_list:
            data = data_from_filepath(filepath).to(device)

            output = model(data)
            predict = output.cpu().detach().numpy()[0][0]

            labels.write("%s %.06f\n" % (filepath, predict))


def main():
    parser = argparse.ArgumentParser(description="Image to Value Regressor")
    subparsers = parser.add_subparsers(help="commands")

    train_parser = subparsers.add_parser("train", help="train model")
    train_parser.add_argument("--dataset", type=str, metavar="DS",
                              help="name of dataset to use")
    train_parser.add_argument("--dataroot", type=str, metavar="DR",
                              help="path of dataroot to use (overrides --dataset)")
    train_parser.add_argument("--name", type=str, metavar="N",
                              help="name of module to create")
    train_parser.add_argument("--epochs", type=int, default=10, metavar="E",
                              help="number of epochs to train (default: 10)")
    train_parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
                              help="learning rate (default: 0.001)")
    train_parser.add_argument("--momentum", type=float, default=0.9, metavar="M",
                              help="SGD momentum (default: 0.9)")
    train_parser.add_argument("--no-cuda", action="store_true", default=False,
                              help="disables CUDA training")
    train_parser.add_argument("--seed", type=int, default=1, metavar="S",
                              help="random seed (default: 1)")
    train_parser.add_argument("--log_interval", type=int, default=10, metavar="LI",
                              help="how many batches to wait before logging training status")
    train_parser.add_argument("--save-interval", type=int, default=10, metavar="SI",
                              help="how many epochs to wait before saving training weights")
    train_parser.set_defaults(which="train")

    test_parser = subparsers.add_parser("test", help="test model")
    test_parser.add_argument("--dataset", type=str, metavar="DS",
                             help="name of dataset to use")
    test_parser.add_argument("--dataroot", type=str, metavar="DR",
                             help="path of dataroot to use (overrides --dataset)")
    test_parser.add_argument("--name", type=str, metavar="N",
                             help="name of module")
    test_parser.add_argument("--no-cuda", action="store_true", default=False,
                             help="disables CUDA testing")
    test_parser.add_argument("--seed", type=int, default=1, metavar="S",
                             help="random seed (default: 1)")
    test_parser.add_argument("--which_epoch", type=int, metavar="WE",
                             help="which epoch to use")
    test_parser.set_defaults(which="test")

    args = parser.parse_args()
    print(args)

    # User has not selected command
    if not hasattr(args, "which"):
        parser.print_usage()
        return

    # Train
    if args.which is "train":
        # Ensure existence of args.dataroot
        if not args.dataroot:
            if args.dataset:
                args.dataroot = os.path.join("datasets", args.dataset)
            else:
                train_parser.print_help()
                print("Error: No dataroot specified. Use --dataset or --dataroot to specify dataroot.", file=sys.stderr)
                return
        if not os.path.exists(args.dataroot):
            print("Error: The dataroot %s could not be found." % args.dataroot)
            return

        # Ensure that name has been specified
        if not args.name:
            train_parser.print_help()
            print("Error: No name specified. Use --name to specify name.", file=sys.stderr)
            return

        # Use cuda if user has not disabled it and is available
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        torch.manual_seed(args.seed)

        dataset_train_path = os.path.join(args.dataroot, "train")
        if not os.path.exists(dataset_train_path):
            dataset_train_path = args.dataroot

        train_loader = load_data(dataset_train_path)

        model = Net().to(device)
        criterion = F.mse_loss
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        train(args, model, device, optimizer, criterion, train_loader)
    elif args.which is "test":
        # Ensure existence of args.dataroot
        if not args.dataroot:
            if args.dataset:
                args.dataroot = os.path.join("datasets", args.dataset)
            else:
                train_parser.print_help()
                print("Error: No dataroot specified. Use --dataset or --dataroot to specify dataroot.", file=sys.stderr)
                return
        if not os.path.exists(args.dataroot):
            print("Error: The dataroot %s could not be found." % args.dataroot)
            return

        # Ensure that name has been specified
        if not args.name:
            train_parser.print_help()
            print("Error: No name specified. Use --name to specify name.", file=sys.stderr)
            return

        # Use cuda if user has not disabled it and is available
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        torch.manual_seed(args.seed)

        dataset_test_path = os.path.join(args.dataroot, "test")
        if not os.path.exists(dataset_test_path):
            dataset_test_path = args.dataroot

        test_list = sorted(glob(os.path.join(dataset_test_path, "*")))

        model_path = os.path.join("models", args.name)
        weights_filepath = os.path.join(model_path, "weights.pth")

        model = Net().to(device)
        weights = torch.load(weights_filepath)
        model.load_state_dict(weights)

        test(args, model, device, test_list)


if __name__ == "__main__":
    main()
