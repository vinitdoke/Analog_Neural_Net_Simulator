import numpy as np
import matplotlib.pyplot as plt

import torch

# TODO see numba optimization


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


# def conv2D_to_MM(kernel2D, fmap2D, stride=1, padding=0):
#     """
#     Adapted from:
#     https://github.com/rmwkwok/transposed_convolution_in_numpy/blob/main/conv2d_transpose_numpy.py

#     Should Return:
#     1. Input Vector
#     2. Weight Matrix
#     3. Output Reshape Dimensions
#     """

#     # comput output dimensions
#     kernel_size = kernel2D.shape[0]
#     fmap_size = fmap2D.shape[0]

#     output_size = (fmap_size - kernel_size + 2 * padding) // stride + 1
#     output_shape = (output_size, output_size)

#     fmap2D = np.pad(fmap2D, padding, mode="constant")
#     input_tensor = fmap2D.flatten()

#     # unroll the kernel
#     kernel_flat = kernel2D.flatten()
#     toeplitz_matrix = np.zeros((output_size**2, len(input_tensor) ))
    
#     for i in range(output_size):
#         for j in range(output_size):
#             toeplitz_matrix[i*output_size + j, i*fmap_size + j : i*fmap_size + j + kernel_size**2] = kernel_flat
    
#     return input_tensor, toeplitz_matrix, output_shape


# def my_convolve2d(a, conv_filter):
#     submatrices = np.array([
#          [a[:-2,:-2], a[:-2,1:-1], a[:-2,2:]],
#          [a[1:-1,:-2], a[1:-1,1:-1], a[1:-1,2:]],
#          [a[2:,:-2], a[2:,1:-1], a[2:,2:]]])
#     multiplied_subs = np.einsum('ij,ijkl->ijkl',conv_filter,submatrices)
#     return np.sum(np.sum(multiplied_subs, axis = -3), axis = -3)


# def im2col(I, K_shape):
#     H, W = I.shape
#     kH, kW = K_shape
#     out_h = H - kH + 1
#     out_w = W - kW + 1
    
#     col = np.zeros((out_h * out_w, kH * kW))
    
#     for y in range(out_h):
#         for x in range(out_w):
#             patch = I[y:y + kH, x:x + kW].flatten()
#             col[y * out_w + x, :] = patch
            
#     return col

# def conv2D_to_MM(I, K):
#     I_flat = I.flatten()
#     B = im2col(I, K.shape)

#     print(B.shape)
#     return I_flat, B, (I.shape[0] - K.shape[0] + 1, I.shape[1] - K.shape[1] + 1)


def serial_conv2D_to_VMM(fmap, kernels, stride=1, padding=0):
    pass
    

if __name__ == "__main__":
    pass

    # np.random.seed(0)

    # ## check quantize

    # clip_range = [-1, 1]
    # quadrature_delta = 0.1

    # weights = np.random.randn(1000)
    # quantized = quantize(weights, quadrature_delta, clip_rangqe)

    # quant_unique, quant_counts = np.unique(quantized, return_counts=True)

    # print(quant_unique)
    # print(quant_counts)

    # plt.stem(quant_unique, quant_counts, label="quantized", markerfmt="ro")
    # plt.hist(weights, label="Weight Dist", alpha=0.5)

    # plt.title(f"Quadrature Delta: {quadrature_delta} Clip Range: {clip_range}")
    # plt.legend()
    # plt.show()

    # check weights_to_differential_conductance

    # weights = np.random.randn(3, 3)
    # conductance_matrix = weights_to_differential_conductance(weights)

    # print(weights)
    # print(conductance_matrix)



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
