
import torch
import matplotlib.pyplot as plt
import numpy as np

from petws.src import dataset, utils

DATA = dataset.HDF5Dataset(which_patients_indices=range(400, 499), resize=True, return_class_label=False ) # 10, 13


loader = torch.utils.data.DataLoader(DATA, batch_size=1, num_workers=2, shuffle=False)


def threshold(y_pred, threshold_value=0):
    """Threshold for BCE Classification
    Args:
        y_pred (array): Predictions
    Returns:
        binary prediction
    """
    prediction = (y_pred > threshold_value) * 1
    return prediction


def run( unet, store_result, device):

    model = unet

    i = 0

    ious = []
    accs = []
    dices = []
    unet_segmentation = np.zeros((len(DATA), 256, 256))

    for data, label in loader:

        if i % 5000 == 0:
            print(i)

        pred = model(data.to(device))
        pred = threshold(pred.detach().cpu().numpy())[0]
        segmentation = label[0]
        segmentation = segmentation.numpy()[0]

        segmentation_map_of_roi = pred
        segmentation_map_of_roi = segmentation_map_of_roi.astype(np.float16)
        dice = utils.dice(pred, segmentation)
        dices.append(dice)

        if store_result:
            unet_segmentation[i] = segmentation_map_of_roi.astype(np.uint8)
        i+=1
    print(np.mean(dices), np.median(dices))

    with open("Res_UNET_SUPERVISED_TEST.txt", "a") as f:
        f.write(f'{np.mean(dices)}, {np.median(dices)}')
        f.write("\n")

    if store_result:
        np.save("supervised_test_segmentation", unet_segmentation)