from __future__ import print_function
import argparse
import cv2
import numpy as np
import os
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


def load_data(filename):
    loader = []
    dirname = os.path.dirname(filename)

    with open(filename, "r") as f:
        contents = f.readlines()
        for line in contents:
            line = line.strip().split()
            print(line)

            (data, target) = (None, None)

            if len(line) >= 1:
                data = cv2.resize(cv2.imread(os.path.join(dirname, line[0])), (32, 32))
                data = data[:, :, ::-1]
                data = np.swapaxes(np.swapaxes(np.array(data, dtype=float), 0, 2), 1, 2) / 255.0
                data_shape = (1,) + data.shape
                data = torch.from_numpy(data.reshape(data_shape)).float()

            if len(line) == 2:
                target = torch.tensor([[float(line[1])]])

            loader.append((data, target))

    return loader


def train(args, model, device, optimizer, criterion, train_loader):
    model.train()
    module_path = os.path.join("modules", args.name)

    with open(os.path.join(module_path, "train.log"), "w") as log:
        for epoch in range(1, args.epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader, 1):
                data, target = data.to(device), target.to(device)
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
                torch.save(model.state_dict(), os.path.join(module_path, "checkpoints", "save_%06d.pth" % epoch))

        torch.save(model.state_dict(), os.path.join(module_path, "weights.pth"))


def test(args, model, device, test_loader):
    model.eval()

    with open("results/labels.txt", "w") as lol:
        for (data, target) in test_loader:
            data = data.to(device)

            output = model(data)
            predict = output.cpu().detach().numpy()[0][0]

            print(predict)
            lol.write("%f\n" % predict)

            # img = data.cpu().detach().numpy()
            # img = img.reshape(img.shape[1:])
            # img = np.swapaxes(np.swapaxes(np.array(img, dtype=float), 0, 2), 0, 1)
            # img = img[:, :, ::-1]
            # cv2.imshow("img", img)
            # cv2.waitKey()

        # cv2.destroyAllWindows()


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

    if not hasattr(args, "which"):  # User has not selected command
        parser.print_usage()
        return

    if args.which is "train":
        # Ensure existence of args.dataroot
        if not args.dataroot:
            if args.dataset:
                args.dataroot = os.path.join("datasets", args.dataset)
            else:
                train_parser.print_help()
                return
        if not args.name:
            train_parser.print_help()
            return

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(args.seed)

        labels_file = os.path.join(args.dataroot, "train", "labels.txt")
        train_loader = load_data(labels_file)

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
                test_parser.print_help()
                return
        if not args.name:
            test_parser.print_help()
            return

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(args.seed)

        images_file = os.path.join(args.dataroot, "test", "filelist.txt")
        test_loader = load_data(images_file)

        model = Net().to(device)
        weights = torch.load(os.path.join("modules", args.name, "weights.pth"))
        model.load_state_dict(weights)

        test(args, model, device, test_loader)


if __name__ == "__main__":
    main()
