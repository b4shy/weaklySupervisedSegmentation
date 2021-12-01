import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage.exposure import histogram
from skimage.filters import sobel

from petws.src import dataset, utils

DATA = dataset.HDF5Dataset(which_patients_indices=range(400, 499), resize=True, return_class_label=False ) # 10, 13


loader = torch.utils.data.DataLoader(DATA, batch_size=1, num_workers=2, shuffle=False)



def run(threshold_percentile, store_result, model, device):

    i = 0
    dices = []
    threshold_segmentations = np.zeros((len(DATA), 256, 256))

    for data, label in loader:

        if i % 5000 == 0:
            print(i)

        pred = model(data.to(device))
        if pred < 0:
            segmentation_map_of_roi = threshold_segmentations[i]

            if store_result:
                threshold_segmentations[i] = segmentation_map_of_roi.astype(np.uint8)
            i+=1
            continue
        
        pet = data[0][0]
        segmentation = label[0]
        data = pet.numpy()
        segmentation = segmentation.numpy()[0]

        ################# Histogramm ################################
        hist, hist_centers = histogram(data, nbins=20)
        threshold = np.percentile(hist_centers, threshold_percentile)
        #############################################################

        ################# Segmentation based on Threshold############

        segmentation_map_of_roi = data > threshold
        ############################################################
        segmentation_map_of_roi = segmentation_map_of_roi.astype(np.float16)
        dice = utils.dice(segmentation_map_of_roi, segmentation)
        dices.append(dice)
        if store_result:
            threshold_segmentations[i] = segmentation_map_of_roi.astype(np.uint8)
        i+=1
    print(np.mean(dices), np.median(dices))

    with open("Threshold_Test.txt", "a") as f:
        f.write(f'Threshold:{threshold_percentile} \n')
        f.write(f'{np.mean(dices)}, {np.median(dices)}')
        f.write("\n")

    if store_result:
        np.save("threshold_test_35", threshold_segmentations)
    return np.mean(dices), np.median(dices)
