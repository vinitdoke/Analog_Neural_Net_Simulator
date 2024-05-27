"""
Control script for the evaluation of the models, CLI sweep, Logging, and plotting.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from training.model_training import NN, get_loaders


def compare_models(models=[], names=None, max_samples=10):

    if names is None:
        names = [f"Model {i}" for i in range(len(models))]

    test_data = get_loaders(batch_size=1, train=False, dataset_path="datasets/")
    confusion_matrices = [np.zeros((10, 10)) for _ in models]
    total_samples = 0
    models_corrects = [0 for _ in models]
    device = "cpu"

    with torch.no_grad():
        
        for data in test_data:

            if max_samples is not None and total_samples >= max_samples:
                break

            images, labels = data
            images, labels = images.to(device), labels.to(device)
            total_samples += labels.size(0)

            for i, model in enumerate(models):
                print(f"Testing {names[i]}, {total_samples}/{max_samples} Tested", end="\r")

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                models_corrects[i] += (predicted == labels).sum().item()

                confusion_matrices[i][labels, predicted] += 1




    # print accuracy
    for i, model in enumerate(models):
        print(f"{names[i]}: {models_corrects[i] / total_samples}")

    # Plot confusion matrices for each model
    fig, axs = plt.subplots(1, len(models), figsize=(6*len(models), 5))

    # print(len(axs))
    for i, model in enumerate(models):

        if len(models) == 1:
            ax = axs
        else:
            ax = axs[i]
        
        sns.heatmap(confusion_matrices[i], annot=True, fmt='g', ax=ax, cmap="Blues")
        ax.set_title(f"{names[i]}; Accuracy: {models_corrects[i] / total_samples}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        # ax.set_xticks(range(10))
        # ax.set_yticks(range(10))





        # if len(models) == 1:
        #     ax = axs
        # else:
        #     ax = axs[i]
        # ax.imshow(confusion_matrices[i], cmap='hot', interpolation='nearest')
        # ax.set_title(f"{names[i]}; Accuracy: {models_corrects[i] / total_samples}")
        # ax.set_xlabel("Predicted")
        # ax.set_ylabel("True")
        # ax.set_xticks(range(10))
        # ax.set_yticks(range(10))
        # # for j in range(10):
        # #     for k in range(10):
        # #         ax.text(j, k, int(confusion_matrices[i][j, k]), ha="center", va="center", color="black")

    plt.show()






