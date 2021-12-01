import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def threshold(y_pred, threshold_value=0):
    """Threshold for BCE Classification
    Args:
        y_pred (array): Predictions
        threshold_values (int): Threshold value
    Returns:
        binary prediction
    """
    prediction = (y_pred > threshold_value) * 1
    return prediction


def accuracy(y_pred, labels):
    """Computes the accuracy

    Args:
        y_pred (array): Prediction
        labels (array): Class labels
    Returns: int: Accuracy
    """
    threshed_prediction = threshold(y_pred)
    corrects = (threshed_prediction == labels) * 1.0
    return torch.mean(corrects)


def dice(pred, label):
    """Returns the intersction over union given the predicted segmentation and the Label
    Args:
        pred (array): GradCam Segmentation
        label (array): Labeled Segmentation
    Returns:
        int: intersection over union
    """
    intersection = np.logical_and(pred, label)
    union = np.logical_or(pred, label)
    denum = (np.sum(pred) + np.sum(label))
    if denum == 0:
        return 1
    dice = 2*np.sum(intersection) / denum
    
    return dice


def compute_loss_weight(data):
    """Compute the weight for the weighted loss

    Args:
        data (dataset): The training dataset

    Returns:
        float: The weight for the loss
    """
    no_tumor = (data.df["Label"] == 0).sum()
    tumor = (data.df["Label"] == 1).sum()
    print(f'Slices without tumors: {no_tumor}; Slices with tumors: {tumor}')
    assert (no_tumor + tumor) == len(data)
    return no_tumor / tumor

def assert_no_overlap_train_test(train, test):
    """Makes sure that there is no overlap between train and test data

    Args:
        train (dataset): Train Dataset
        test (dataset): Test Dataset
    """
    assert not bool(set(train.df["Key"]) & set(test.df["Key"]))


def load_weights(path, skip_first_characters):
    """Load the weights and map them to the correct network strings

    Args:
        path (string): path to the weights
        skip_first_characters (int): How many characters to skip (eg. vgg.Conv2D -> Conv2D)

    Returns:
        [type]: [description]
    """
    d = torch.load(path, map_location=torch.device("cpu"))
    new_state = {}
    for k, v in d.items():
        new_state[k[skip_first_characters:]] = v
    return new_state





def create_heatmap_figure(pet, ct, mask, prepare_heatmap, inputs,
                          heatmap_resnet, heatmap_resnet_grad, heatmap_resnet_grad_pp,
                          heatmap_vgg, heatmap_vgg_grad, heatmap_vgg_grad_pp, index, path_prefix):

    fig, axis = plt.subplots(2, 4, figsize=(15, 5))
    axis[0][0].imshow(pet, cmap="gray", vmin=0, vmax=0.1)
    prepared_heatmap = prepare_heatmap(heatmap_resnet, inputs.shape)
    axis[0][0].imshow(prepared_heatmap, alpha=0.3)
    axis[0][0].set_title("ScoreCAM ResNET")

    axis[0][1].imshow(pet, cmap="gray", vmin=0, vmax=0.1)
    heatmap_prepared_grad = prepare_heatmap(heatmap_resnet_grad, inputs.shape)
    axis[0][1].imshow(heatmap_prepared_grad, alpha=0.3)
    axis[0][1].set_title("GradCAM ResNET")

    axis[0][2].imshow(pet, cmap="gray", vmin=0, vmax=0.1)
    heatmap_prepared_gradpp = prepare_heatmap(heatmap_resnet_grad_pp, inputs.shape)
    axis[0][2].imshow(heatmap_prepared_gradpp, alpha=0.3)
    axis[0][2].set_title("GradCAM++ ResNET")

    axis[0][3].imshow(pet, cmap="gray", vmin=0, vmax=0.1)
    mask = np.ma.masked_where(mask==0, mask)
    axis[0][3].imshow(mask, alpha=0.4)


    axis[1][0].imshow(pet, cmap="gray", vmin=0, vmax=0.1)
    prepared_heatmap = prepare_heatmap(heatmap_vgg, inputs.shape)
    axis[1][0].imshow(prepared_heatmap, alpha=0.3)
    axis[1][0].set_title("ScoreCAM VGG")

    axis[1][1].imshow(pet, cmap="gray", vmin=0, vmax=0.1)
    heatmap_prepared_grad = prepare_heatmap(heatmap_vgg_grad, inputs.shape)
    axis[1][1].imshow(heatmap_prepared_grad, alpha=0.3)
    axis[1][1].set_title("GradCAM VGG")

    axis[1][2].imshow(pet, cmap="gray", vmin=0, vmax=0.1)
    heatmap_prepared_gradpp = prepare_heatmap(heatmap_vgg_grad_pp, inputs.shape)
    axis[1][2].imshow(heatmap_prepared_gradpp, alpha=0.3)
    axis[1][2].set_title("GradCAM++ VGG")

    axis[1][3].imshow(ct, cmap="gray", vmin=0)

    for i in range(2):
        for j in range(4):
            axis[i][j].axis("off")

    save_path = os.path.join(path_prefix, f"comparison:{index}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close('all')
