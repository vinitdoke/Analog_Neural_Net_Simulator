import torch
import torchvision
import torch.nn.functional as F

from training.model_training import NN, get_accuracy, get_loaders
from interface.IOTransfer import quantize

import numpy as np

dataset_path = "../datasets"


class PseudoHardwareNN(NN):

    def __init__(
        self,
        hardware_multiplier=None,
        input_quantization=(False, 0.1),
        weight_quantization=(False, 0.05, [-1, 1]),
        output_quantization=(False, 0.1),
        verbose=False,
    ):
        super(PseudoHardwareNN, self).__init__()
        self.printed = False
        self.hardware_multiplier = hardware_multiplier

        self.input_quantization = input_quantization
        self.weight_quantization = weight_quantization
        self.output_quantization = output_quantization

    def hardware_propagate(self, layer, x):
        """
        Propagate input through hardware multiplier.
        """

        weights = layer.weight.cpu().numpy()
        x = x.cpu().numpy()

        if self.hardware_multiplier is None:

            if self.printed == False:
                print("No hardware multiplier provided. Using software multiplication.")
                self.printed = True
            x = np.matmul(x, weights.T)

        else:

            # flatten input
            x = x.flatten()

            ## QUANTIZATION
            if self.input_quantization[0]:
                x = quantize(x, self.input_quantization[1])
            if self.weight_quantization[0]:
                weights = quantize(
                    weights, self.weight_quantization[1], self.weight_quantization[2]
                )

            # hardware multiplication
            x = self.hardware_multiplier(x, weights.T).matmul()

            # quantize output
            if self.output_quantization[0]:
                x = quantize(x, self.output_quantization[1])

        # convert back to tensor
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

            x = self.hardware_propagate(layer, x)

            x += layer.bias

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
