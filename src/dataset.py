import os
from pathlib import Path
from collections import deque

import h5py
import dotenv
import nibabel as nib
import numpy as np
import pandas as pd
import tqdm
import torch
from imgaug import augmenters as iaa


dotenv.load_dotenv()
DATA = Path(os.getenv("DATA"))

class HDF5Dataset(torch.utils.data.Dataset):
    """Dataset class for HDF5 data. Converts 3D PET and CT Data to 2D Slices
    """

    def __init__(self, hdf5_path=DATA/'raw/TUE0000ALLDS_3D.h5', df_path=DATA/"raw/2dSlices_df",
                 what_data_to_use=2, method="dict", which_patients_indices=-1, resize=False, return_class_label=True, augment=False):
        """Initializes the Dataset.


        Args:
            hdf5_path (pathlib.Path): path_to_the_data
            df_path (pathlib.Path): path to the dataframe
            what_data_to_use (string or int): use either only PET or PET+CT or PET+CT+MASK.
             Use 0 ("PET") for only PET data
             Use 1 ("CT") for PET and CT data
             Use 2 ("MASK") for PET and CT and MASK data
            method (string): Which method to use: "dict" to use a dict which contains all data. "deque" to use a deque
            paired with a search list
            which_patiens_indices (list or int): Which patiens to use for this dataset (-1) for all. Pass a list or range
            resize (bool): Pad or crop all images to the same size
            return_class_label (bool): Return class label or mask

        """

        self.hdf5_path = hdf5_path
        self.df = pd.read_feather(df_path)
        self.what_data_to_use = what_data_to_use
        self.resize = resize
        self.return_class_label = return_class_label    

        keys = HDF5Dataset.compute_keys(hdf5_path)
        self.subset = HDF5Dataset.set_subset(keys, which_patients_indices)
        keys_subset = np.array(keys)[self.subset]
        self.method = method
        
        self.data = self.create_data_via_method()
        self.augment = augment
        self.augmenter_affine = iaa.Sequential([iaa.Affine(scale=(0.85, 1.15), rotate=(-10, 10), translate_percent=(-0.1, 0.1)),
                                        ])
        self.augmenter_contrast = iaa.contrast.GammaContrast()


        self.length  = HDF5Dataset.compute_length(self.data)

        self.df = self.df[self.df["Key"].isin(keys_subset)]
        assert self.length == len(self.df)

    @staticmethod
    def load_data_in_dict(path, subset, what_data_to_use):
        """Loads the data from the hdf5 container and converts it into a dictionary of the form:
        {key: array of the shape (x, h, w, s)} where x depends on what_data_to_use (0:PET, 1:CT, 2:MASK)

        Args:
            subset (int): Use only a subset of the data. -1 for all data
            what_data_to_use (int/string): What data to use. 0/PET for only PET, 1/CT for PET and CT, 2/MASK for PET/CT/MASK

        Returns:
            dict: dictionary containing the data for all patients with the key as index
        """
        all_data = {}
        with h5py.File(path, 'r') as hf:
            keys = np.array(list(hf['image']))
            print(keys[subset])

            for key in tqdm.tqdm(keys[subset]):
                data = hf[f'image/{key}']
                if what_data_to_use == 0 or what_data_to_use == "PET":
                    pet = data[0]
                    all_data[key] = np.array([pet])

                elif what_data_to_use == 1 or what_data_to_use == "CT":
                    pet = data[0]
                    ct = data[1]
                    all_data[key] = np.array([pet, ct])

                elif what_data_to_use == 2 or what_data_to_use == "MASK":
                    pet = data[0]
                    ct = data[1]
                    mask = hf[f'mask_iso/{key}'][0]
                    pet_ct_mask = np.array([pet, ct, mask])
                    all_data[key] = pet_ct_mask
        return all_data

    @staticmethod
    def generator(path, subset, what_data_to_use):
        """Creates generator containing the patient data of the shape(x, h, w, s). Again x depends on used_data
           In order to match a patient to the elements in the arrays, it needs to be paired with the search_dictionary

        Args:
            path (string): path to the h5 data
            subset (int): Use only a subset of the data. -1 for all data
            what_data_to_use (int/string): What data to use. 0/PET for only PET, 1/CT for PET and CT, 2/MASK for PET/CT/MASK

        Yields:
            array: Array containing the data
        """
        with h5py.File(path, 'r') as hf:
            keys = np.array(list(hf['image']))
            for key in tqdm.tqdm(keys[subset]):
                data = hf[f'image/{key}']
                mask = hf[f'mask_iso/{key}'][0]
                project = data.attrs["project"]

                if what_data_to_use == 0 or what_data_to_use == "PET":
                    pet = data[0]
                    yield np.array([pet])

                elif what_data_to_use == 1 or what_data_to_use == "CT":
                    pet = data[0]
                    ct = data[1]
                    yield np.array([pet, ct])

                elif what_data_to_use == 2 or what_data_to_use == "MASK":
                    pet = data[0]
                    ct = data[1]
                    mask = hf[f'mask_iso/{key}'][0]
                    yield np.array([pet, ct, mask])
                    
    @staticmethod
    def create_search_dictionary(path, subset):
        """Creates a search dictionary which maps a given key to an index, corresponding the position in the generator
        Args:
            path (string): path to the h5 data
            subset (int): Use only a subset of the data. -1 for all data

        Returns:
            dict: lookup dictionary of the form {key: index in the generator}
        """
        search_dict = {}
        patient_counter = 0
        with h5py.File(path, 'r') as hf:
            keys = np.array(list(hf['image']))
            for key in keys[subset]:
                data = hf[f'image/{key}']
                search_dict[key] = patient_counter
                patient_counter+=1
        return search_dict

    def create_data_via_method(self):
        """Creates the data container corresponding to the desired method (deque or dictionary). If dict is used, then
           a dictionary of the form {key: array of the shape (x, h, w, s)} is returned. If deque is used, then a deque
           where each entry corresponds to the data of the patient is returned, combined with the lookup dictionary

        Returns:
            dict or deque: Either data dict or deque dict
        """
        if self.method == "dict":
            data = HDF5Dataset.load_data_in_dict(self.hdf5_path, self.subset, self.what_data_to_use)
            return data
        if self.method == "deque":
            generator = HDF5Dataset.generator(self.hdf5_path, self.subset, self.what_data_to_use)
            self.search_dict = HDF5Dataset.create_search_dictionary(self.hdf5_path, self.subset)
            data = deque(generator)
            return data
            
    @staticmethod
    def compute_length(data):
        """Computes the number of 2D slices in the dataset. Should correspond to the number of rows in the table

        Args:
            data ([dict or deque]): Data container

        Returns:
            int: Number of 2D slices
        """
        length=0
        try:
            for i in data:
                length+=i.shape[-1]
        except IndexError:
            for i in data.values():
                length+=i.shape[-1]

        return length


    @staticmethod
    def compute_keys(path):
        """Returns all keys in the dataset

        Args:
            path (Pathlib path): Path to the dataset

        Returns:
            list: Keys in the dataset
        """
        with h5py.File(path, 'r') as hf:
            keys = list(hf['image'])
        return keys

    @staticmethod
    def set_subset(keys, which_patients_indices):
        """Sets the subset variable to -1 if all data shall be used or to the corresponding subset

        Args:
            use_subset (bool or int): False or -1 if all data shall be used

        Returns:
            int: The subset size
        """
        if which_patients_indices == -1:
            subset = range(len(which_patients_indices))
        else:
            subset = which_patients_indices
        return subset


    def __len__(self):
        """Return the number of 2D slices in the dataset

        Returns:
            int: number of 2D slices
        """
        return self.length


    @staticmethod
    def pad_or_crop(data, label):
        padd_aug = iaa.size.CenterPadToFixedSize(256, 256)
        crop_aug = iaa.size.CenterCropToFixedSize(256, 256)
        reshaped_slices = np.rollaxis(data, 0, 3)

        if data.shape[1] < 256:
            data_aug = padd_aug(image=reshaped_slices)
            label_aug = padd_aug(image=label)
        
        else:
            data_aug = crop_aug(image=reshaped_slices)
            label_aug = crop_aug(image=label)

        data_aug = np.rollaxis(data_aug, 2, 0)

        return data_aug, label_aug

    def augment_slices_and_labels(self, data, label):
        """Augment slices and labels. Only apply contrast changes to PET
        Args:
            data (array): Unaugmented slice
            label (array): Unaugmented segmentation
        """
        reshaped_slices = np.rollaxis(data, 0, 3)

        _aug = self.augmenter_affine._to_deterministic()
        augmented_slices = _aug(image=reshaped_slices)
        augmented_slices[:,:,0] = self.augmenter_contrast(image=augmented_slices[:,:,0])
        
        try:
            augmented_segmentation = _aug(image=label)
            augmented_segmentation[augmented_segmentation > 0] = 1
        except AttributeError:
            augmented_segmentation = label
        augmented_slices = np.rollaxis(augmented_slices, 2, 0)

        return augmented_slices, augmented_segmentation


    def __getitem__(self, idx):
        """Returns the data

        Args:
            idx (int): Return this slice

        Returns:
            array: Data array containing either only data or petct + label/mask
        """

        key, project, position, label, slice_ = self.df.iloc[idx]

        if self.method == "deque":
            data= self.data[self.search_dict[key]][:,:,:,position]

        elif self.method == "dict":
            data = self.data[key][:,:,:,position]
        

        if data.shape[0] == 3:
            data_ = data[:2, :, :].astype(np.float32)
            label = data[2,:,:].astype(np.float32)

            if self.resize:
                data_, label = HDF5Dataset.pad_or_crop(data_, label)
            
            if self.return_class_label:
                label = int(label.sum() > 0)
            
            if self.augment:
                data_, label = self.augment_slices_and_labels(data_, label)

            return data_, np.array(np.expand_dims(label, 0), dtype=np.float32)
        else:
            return data