import numpy as np
import matplotlib.pyplot as plt


def quantize(x, delta, clip_range=None):
    """
    If clip range is not provided, the function will use the min and max of the input x as the clip range.
    """
    if clip_range is None:
        clip_range = [np.min(x), np.max(x)]

    bins = np.arange(clip_range[0], clip_range[1], delta)
    return bins[np.digitize(np.clip(x, *clip_range), bins) - 1]


def weights_to_differential_conductance(weight_matrix):

    """
    This function takes in a weight matrix (m,n) and returns a differential conductance matrix(m,n,2)
    """

    





    return None


def conv2D_to_VMM(filter2D, feature_map2D):
    pass


if __name__ == "__main__":

    np.random.seed(0)

    clip_range = [-0.5, 1]
    quadrature_delta = 0.1

    weights = np.random.randn(1000)
    quantized = quantize(weights, quadrature_delta, clip_range)

    quant_unique, quant_counts = np.unique(quantized, return_counts=True)

    print(quant_unique)
    print(quant_counts)

    plt.stem(quant_unique, quant_counts, label="quantized", markerfmt="ro")
    plt.hist(weights, label="Weight Dist", alpha=0.5)

    plt.title(f"Quadrature Delta: {quadrature_delta} Clip Range: {clip_range}")
    plt.legend()
    plt.show()
