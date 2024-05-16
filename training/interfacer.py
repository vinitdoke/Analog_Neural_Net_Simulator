import torch
import torchvision
import torch.nn.functional as F

from training.model_training import NN, get_accuracy, get_loaders

import numpy as np

dataset_path = "../datasets"


class PseudoHardwareNN(NN):

    def __init__(self, hardware_multiplier=None):
        super(PseudoHardwareNN, self).__init__()
        self.printed = False
        self.hardware_multiplier = hardware_multiplier

    def hardware_propagate(self, layer, x):

        weights = layer.weight.cpu().numpy()
        x = x.cpu().numpy()

        if self.hardware_multiplier is None:

            if self.printed == False:
                print("No hardware multiplier provided. Using software multiplication.")
                self.printed = True
            # perform matrix multiplication
            x = np.matmul(x, weights.T)

        else:
            # perform hardware multiplication

            # flatten input
            x = x.flatten()
            x = self.hardware_multiplier(x, weights.T).matmul()

        # convert back to torch tensor
        # x = torch.tensor(x).to(device=layer.weight.device)
        x = (
            torch.tensor(x)
            .reshape(-1, layer.weight.shape[0])
            .to(device=layer.weight.device)
        )

        return x

    def forward(self, x):
        x = torch.flatten(x, 1)

        model_depth = 3  # fix
        for i, layer in enumerate(self.children()):
            # hardware mutliplication
            x = self.hardware_propagate(layer, x)

            # software bias addition
            x += layer.bias

            # software activation
            if i < model_depth - 1:  # no activation on the last layer
                x = torch.nn.functional.relu(x)

        return x


if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # load model
    model = PseudoHardwareNN()
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)
    test_data = get_loaders(train=False, batch_size=32)
    accuracy = get_accuracy(model, test_data, DEVICE="mps")
    print(f"Accuracy: {accuracy}")
