import torch
import torchvision
import torch.nn.functional as F

import numpy as np

dataset_path = "../datasets"

### DEFAULT HYPERPARAMETERS ###
batch_size = 64
learning_rate = 1e-3
epochs = 25
#######################


class NN(torch.nn.Module):
    """
    784 -> 16 -> 16 -> 10
    """

    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = torch.nn.Linear(784, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # output = F.log_softmax(x, dim=1)
        return x


def train_loop(
    optimizer,
    loss_fn,
    model,
    train_loader,
    device="cpu",
    returnLoss=False,
    verbose=False,
):
    size = len(train_loader.dataset)

    training_epochLoss = []

    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # add l1 regularization
        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm

        training_epochLoss.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if verbose:
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    if returnLoss:
        return np.mean(np.array(training_epochLoss))


def get_loaders(train=True, batch_size=32, dataset_path="../datasets"):
    if train:
        train_data = torchvision.datasets.MNIST(
            root=dataset_path,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )

        return torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
    else:
        test_data = torchvision.datasets.MNIST(
            root=dataset_path,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )

        return torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True
        )


def train(device="cpu"):

    train_data = torchvision.datasets.MNIST(
        root=dataset_path,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    model = NN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for t in range(epochs):
        train_loop(optimizer, loss_fn, model, train_loader, device=device, verbose=True)
        print(f"Epoch {t+1}")

    # freeze the model
    model.eval()
    return model


def save_model(model):
    torch.save(model.state_dict(), "model_l1reg.pth")
    print("Model saved")


def get_accuracy(model, test_data, DEVICE="cpu", max_samples=None, verbose=False):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data:
            if max_samples is not None and total >= max_samples:
                break

            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            if verbose:
                print(f"Total Checked: {total}", end="\r")

            correct += (predicted == labels).sum().item()
    return correct / total


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = train(device=device)
    save_model(model)
    test_data = get_loaders(train=False, batch_size=32)
    accuracy = get_accuracy(model, test_data, DEVICE="mps")
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
