import numpy as np
import matplotlib.pyplot as plt

import torch


def quantize(x, delta, clip_range=None):
    """
    If clip range is not provided, the function will use the min and max of the input x as the clip range.
    If all the values in x are the same, the function will return the input x.
    """
    if clip_range is None:
        clip_range = [np.min(x), np.max(x)]

    if clip_range[0] == clip_range[1]:
        return x

    bins = np.arange(clip_range[0], clip_range[1], delta)
    return bins[np.digitize(np.clip(x, *clip_range), bins) - 1]


def weights_to_differential_conductance(weight_matrix, G_OFF=1e-9):
    """
    This function takes in a weight matrix (m,n) and returns a differential conductance matrix(m,n,2)
    (:,:,0) is the positive conductance and (:,:,1) is the negative conductance.
    """

    m, n = weight_matrix.shape
    conductance_matrix = np.zeros((m, n, 2))

    positive_mask = np.where(weight_matrix > 0)
    negative_mask = np.where(weight_matrix <= 0)

    conductance_matrix[:, :, 0] = abs(weight_matrix)
    conductance_matrix[:, :, 1] = abs(weight_matrix)
    conductance_matrix[positive_mask[0], positive_mask[1], 1] = (
        G_OFF  # positive weights are off on the negative side
    )
    conductance_matrix[negative_mask[0], negative_mask[1], 0] = (
        G_OFF  # negative weights are off on the positive side
    )

    return conductance_matrix


def serial_conv2D_to_VMM(fmap, kernels, stride=1, padding=0):
    pass


if __name__ == "__main__":

    np.random.seed(0)

    # ## check quantize

    # clip_range = [-1, 1]
    # quadrature_delta = 0.1

    # weights = np.random.randn(1000)
    # quantized = quantize(weights, quadrature_delta, clip_range)

    # quant_unique, quant_counts = np.unique(quantized, return_counts=True)

    # # print(quant_unique)
    # # print(quant_counts)

    # plt.stem(quant_unique, quant_counts, label="Quantized and Clipped", markerfmt="ro")
    # plt.hist(weights, label="Weight Dist", alpha=0.5)

    # plt.title(f"Quadrature Delta: {quadrature_delta} Clip Range: {clip_range}", fontsize=12)
    # plt.xlabel("Weight Value", fontsize=12)
    # plt.ylabel("Count", fontsize=12)
    # plt.figsize = (10, 10)
    # plt.grid()
    # plt.legend()

    # #rasterize

    # plt.savefig("../quantization.png", dpi = 600)

    # plt.show()

    # check weights_to_differential_conductance

    weights = np.random.randn(3, 3)
    conductance_matrix = weights_to_differential_conductance(weights)

    print(weights)
    print(conductance_matrix[:,:,0])
    print(conductance_matrix[:,:,1])

    ## check conv2D_to_MM

    # kernel_size = 3
    # fmap_size = 5

    # kernel = np.identity(kernel_size)
    # fmap = np.arange(fmap_size**2).reshape(fmap_size, fmap_size)

    # print(kernel)
    # print(fmap)

    # input_tensor, toeplitz_matrix, output_shape = conv2D_to_MM(fmap, kernel)

    # print(input_tensor)
    # print(toeplitz_matrix)
    # print(output_shape)

    # # print(my_convolve2d(fmap, kernel))

    # print(np.matmul(toeplitz_matrix, input_tensor).reshape(output_shape))
