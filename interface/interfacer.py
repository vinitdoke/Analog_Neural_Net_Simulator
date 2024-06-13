import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module

from training.model_training import NN, get_accuracy, get_loaders
from training.cnn import CNN, get_loaders_cnn
from interface.IOTransfer import quantize, weights_to_differential_conductance
from hardware.crossbars import DifferentialCrossbar

import numpy as np

dataset_path = "../datasets"


class PseudoHardwareNN(NN):

    def __init__(
        self,
        hardware_multiplier=None,
        input_quantization=(False, 0.1),
        weight_quantization=(False, 0.05, [-1, 1]),
        output_quantization=(False, 0.1),
        weight_variance=None,
        inline_resistances=(False, (0, 0)),
        verbose=False,
    ):
        super(PseudoHardwareNN, self).__init__()
        self.printed = False
        self.verbose = verbose
        self.hardware_multiplier = hardware_multiplier

        ## NON-IDEALITIES
        # quantization
        self.input_quantization = input_quantization
        self.weight_quantization = weight_quantization
        self.output_quantization = output_quantization

        # weight variance
        self.weight_variance = weight_variance

        # in-line resistances
        self.inline_resistances = inline_resistances

        ## Instantiate hardware multipliers later
        self.multipliers = None

    def init_multipliers(self):

        self.multipliers = []

        with torch.no_grad():
            for i, layer in enumerate(self.children()):

                weights = layer.weight.cpu().numpy()
                # print(i, weights.shape)

                ## Handle differential crossbar
                if self.hardware_multiplier == DifferentialCrossbar:
                    weights = weights_to_differential_conductance(weights.T)
                else:
                    weights = weights.T

                ## Quantize weights
                if self.weight_quantization[0]:
                    weights = quantize(
                        weights,
                        self.weight_quantization[1],
                        self.weight_quantization[2],
                    )

                ## Add weight variance
                if self.weight_variance[0]:
                    weights += np.random.normal(
                        0, self.weight_variance[1], weights.shape
                    )

                ## Create hardware multiplier instances
                if self.inline_resistances[0]:
                    self.multipliers.append(
                        self.hardware_multiplier(
                            weights,
                            inline_resistances=self.inline_resistances[1],
                            verbose=self.verbose,
                        )
                    )
                else:
                    self.multipliers.append(
                        self.hardware_multiplier(weights, verbose=self.verbose)
                    )

    def fast_hw_propagate(self, i, x):
        x = x.cpu().numpy()
        x = x.flatten()

        ## Quantize input
        if self.input_quantization[0]:
            x = quantize(x, self.input_quantization[1])

        ## Hardware multiplication
        x = self.multipliers[i].matmul(x)

        ## Quantize output
        if self.output_quantization[0]:
            x = quantize(x, self.output_quantization[1])

        return x

        # def hardware_propagate(self, layer, x):
        """
        Propagate input through hardware multiplier.
        #TODO : no need to perform quantisation steps for every call
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

            # handle differential crossbar
            if self.hardware_multiplier == DifferentialCrossbar:
                weights = weights_to_differential_conductance(weights.T)
            else:
                weights = weights.T

            # quantize input and weights
            if self.input_quantization[0]:
                x = quantize(x, self.input_quantization[1])

            if self.weight_quantization[0]:
                weights = quantize(
                    weights, self.weight_quantization[1], self.weight_quantization[2]
                )

            # add weight variance
            if self.weight_variance is not None:
                weights += np.random.normal(0, self.weight_variance, weights.shape)

            # hardware multiplication

            if self.inline_resistances[0]:
                x = self.hardware_multiplier(
                    weights, x, inline_resistances=self.inline_resistances[1]
                ).matmul()
            else:
                x = self.hardware_multiplier(weights, x).matmul()

            # quantize output
            if self.output_quantization[0]:
                x = quantize(x, self.output_quantization[1])

        return x

    def forward(self, x):

        if self.hardware_multiplier is not None:
            if self.multipliers is None:
                self.init_multipliers()

        x = torch.flatten(x, 1)

        model_depth = 3  # fix

        weight_list = []
        for i, layer in enumerate(self.children()):

            if self.hardware_multiplier is not None:
                x = self.fast_hw_propagate(i, x)
            else:
                x = x.cpu().numpy()
                x = x.flatten()

                ## Quantize input
                if self.input_quantization[0]:
                    x = quantize(x, self.input_quantization[1])

                if i+1 > len(weight_list):

                    ## Quantize weights
                    if self.weight_quantization[0]:
                        weights = layer.weight.cpu().numpy()
                        weights = quantize(
                            weights, self.weight_quantization[1], self.weight_quantization[2]
                        )

                    ## Weight variance
                    weights = layer.weight.cpu().numpy()
                    if self.weight_variance[0]:
                        weights += np.random.normal(
                            0, self.weight_variance[1], weights.shape
                        )
                    
                    weight_list.append(weights)

                x = np.matmul(x, weight_list[i].T)

                ## Quantize output
                if self.output_quantization[0]:
                    x = quantize(x, self.output_quantization[1])

            # convert back to tensor
            x = (
                torch.tensor(x)
                .reshape(-1, layer.weight.shape[0])
                .to(device=layer.weight.device)
            )

            x += layer.bias

            if i < model_depth - 1:  # no activation on the last layer
                x = torch.nn.functional.relu(x)

        return x


class PseudoHardwareCNN(CNN):

    def __init__(
        self,
        input_quantization=(False, 0.1),
        weight_quantization=(False, 0.05, [-1, 1]),
        output_quantization=(False, 0.1),
        weight_variance=None,
    ):

        super(PseudoHardwareCNN, self).__init__()

        ## NON-IDEALITIES
        # quantization
        self.input_quantization = input_quantization
        self.output_quantization = output_quantization

        # weight tranfer
        self.weight_quantization = weight_quantization
        self.weight_variance = weight_variance
        self.weight_transfer_init = False

    def trial(self):

        for param in self.named_parameters():
            print(param[0], param[1].shape)

    def quantize(self, x, delta, clip_range=None, npy=False):
        if not npy:
            if clip_range is None:
                clip_range = [torch.min(x).item(), torch.max(x).item()]

            if clip_range[0] == clip_range[1]:
                return x

            bins = torch.arange(clip_range[0], clip_range[1], delta, device=x.device)
            return bins[torch.bucketize(torch.clamp(x, *clip_range), bins) - 1]
        else:
            if clip_range is None:
                clip_range = [np.min(x), np.max(x)]

            if clip_range[0] == clip_range[1]:
                return x

            bins = np.arange(clip_range[0], clip_range[1], delta)
            return bins[np.digitize(np.clip(x, *clip_range), bins) - 1]

    def init_weight_tranfer(self):

        for param in self.named_parameters():

            if "weight" in param[0]:
                weights = param[1].cpu().numpy()

                # if self.weight_quantization[0]:
                #     weights = self.quantize(weights, self.weight_quantization[1], self.weight_quantization[2], npy=True)

                if self.weight_variance[0]:
                    weights += np.random.normal(
                        0, self.weight_variance[1], weights.shape
                    )

                self.state_dict()[param[0]] = torch.tensor(weights).to(
                    device=param[1].device
                )

    def forward(self, x):

        if not self.weight_transfer_init and (
            self.weight_quantization[0] or self.weight_variance[0]
        ):
            # print("Initialising weight transfer")
            self.init_weight_tranfer()
            self.weight_transfer_init = True

        # 2 covolutions
        for i in range(2):
            # input quantization
            if self.input_quantization[0]:
                x = self.quantize(x, self.input_quantization[1])

            # hardware step
            if i == 0:
                x = self.conv1(x)
            else:
                x = self.conv2(x)

            # output quantization
            if self.output_quantization[0]:
                x = self.quantize(x, self.output_quantization[1])

            # relu
            x = F.relu(x)

            # max pooling
            x = F.max_pool2d(x, 2, 2)

        # flatten
        x = torch.flatten(x, 1)

        # 3 fully connected layers


        for i in range(3):

            # input quantization
            if self.input_quantization[0]:
                x = self.quantize(x, self.input_quantization[1])

            # hardware step
            if i == 0:
                x = self.fc1(x)
            elif i == 1:
                x = self.fc2(x)
            else:
                x = self.fc3(x)

            # output quantization
            if self.output_quantization[0]:
                x = self.quantize(x, self.output_quantization[1])

            # relu
            if i < 2:  # no activation on the last layer
                x = F.relu(x)

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
