from argparse import ArgumentParser
import matplotlib.pyplot as plt

import numpy as np

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F

"""
Convolutional Neural Network for CIFAR10
"""

dataset_path = "../datasets/"

### DEFAULT HYPERPARAMETERS ###
batch_size = 4
learning_rate = 0.001
epochs = 3
#######################


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)

        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_loop(
    model, loss_fn, optimizer, epochs, train_loader, verbose=True, device="cpu"
):
    size = len(train_loader.dataset)
    epochWiseLoss = []

    for epoch in range(epochs):
        training_epochLoss = []

        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            training_epochLoss.append(loss.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(X)
                if verbose:
                    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

        print(f"Epoch {epoch+1}: Avg. Loss: {np.mean(training_epochLoss):.4f}")
        epochWiseLoss.append(np.mean(training_epochLoss))
        
    return model, epochWiseLoss


def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=batch_size,
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=epochs,
        help="number of epochs to train (default: 10)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=learning_rate,
        help="learning rate (default: 1e-3)",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=dataset_path,
        help="path to dataset (default: ../datasets/)",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="path to save model (default: model.pth)",
    )

    return parser.parse_args()


def view_image(image, label):
    print("Label: ", label)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()


def get_loaders_cnn(train=True, batch_size=batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    if train:
        # CIFAR10
        train_data = torchvision.datasets.CIFAR10(
            root=dataset_path,
            train=True,
            transform=transform,
            download=True,
        )

        return torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
    else:
        test_data = torchvision.datasets.CIFAR10(
            root=dataset_path,
            train=False,
            transform=transform,
            download=True,
        )

        return torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True
        )


if __name__ == "__main__":

    device = "mps"

    args = parse()
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    dataset_path = args.dataset_path
    model_path = args.model_path

    train_loader = get_loaders_cnn(train=True)
    test_loader = get_loaders_cnn(train=False)

    model = CNN().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training
    model, epochLoss = train_loop(
        model, loss_fn, optimizer, epochs, train_loader, verbose=True, device=device
    )

    # Testing
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += len(y)
        print(f"Test Accuracy: {correct/total:.3f}")

    # save model
    torch.save(model.state_dict(), f"cnn_weights.pth")

