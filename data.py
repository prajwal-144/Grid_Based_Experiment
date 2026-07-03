# import torch
# import numpy as np
#
# class LensingDataset(torch.utils.data.Dataset):
#     def __init__(self, directory, classes, num_samples):
#         """
#         The dataset class
#
#         :param directory: Path to the dataset directory
#         :param classes: List of lensing image classes
#         :param num_samples: Number of images in the dataset
#         """
#         super(LensingDataset, self).__init__()
#         self.directory = directory
#         self.classes = classes
#         self.num_samples = num_samples
#     def __len__(self):
#         """
#         :return: Returns the length of the dataset
#         """
#         return self.num_samples*len(self.classes)
#
#     def __getitem__(self, index):
#         """
#         Supplies LR images
#
#         :param index: Index in the dataset to look for
#         :return: LR image, min-max normalized
#         """
#         selected_class = self.classes[index//self.num_samples]
#         class_index = index%self.num_samples
#         # image = torch.tensor(np.array([np.load(self.directory+selected_class+'/sim_%d.npy'%(class_index), allow_pickle=True)],  dtype=np.float32))
#         image = torch.tensor(np.load(self.directory + selected_class + '/sim_%d.npy' % (class_index), allow_pickle=True).astype(np.float32))
#         image = (image - torch.min(image))/(torch.max(image)-torch.min(image))
#         return image


import torch
import numpy as np
import os

EPS = 1e-8

class LensingDataset(torch.utils.data.Dataset):
    def __init__(self, directory, classes, num_samples):
        """
        :param directory: Path to the dataset directory (ends with '/')
        :param classes: List of lensing image class subfolders, e.g., ['no_sub']
        :param num_samples: Number of images per class expected (loader uses sim_0..sim_{num_samples-1})
        """
        super(LensingDataset, self).__init__()
        self.directory = directory if directory.endswith(os.sep) else directory + os.sep
        self.classes = classes
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples * len(self.classes)

    def _load_np_file(self, path):
        # Load allowing pickles (some files are object arrays)
        arr = np.load(path, allow_pickle=True)

        # If arr is an object array that wraps a single object, unwrap it
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
            arr = arr.item()

        # If arr is an object array with two elements (axion case: [image, mass]),
        # extract the image and ignore the second element here.
        # If you want to use the mass later, return it or store it.
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 2:
            # Prefer first element as image (arr[0])
            image_part = arr[0]
            # optional: mass = arr[1]
            arr = image_part

        # Now convert to numeric ndarray
        try:
            arr = np.asarray(arr, dtype=np.float32)
        except Exception as e:
            # Provide helpful error if we still cannot convert
            raise ValueError(f"Cannot convert loaded object from {path} to numeric ndarray: {e}")

        return arr

    def __getitem__(self, index):
        selected_class = self.classes[index // self.num_samples]
        class_index = index % self.num_samples
        path = self.directory + selected_class + '/sim_%d.npy' % class_index

        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected file not found: {path}")

        arr = self._load_np_file(path)

        # Ensure arr is now numeric and shape is either (H, W) or (1, H, W) or (C, H, W)
        if arr.ndim == 2:
            # single channel image -> add channel axis
            arr = arr[None, ...]
        elif arr.ndim == 3:
            # could be (1, H, W) or (C, H, W) -> fine
            pass
        else:
            raise ValueError(f"Unexpected array shape for {path}: {arr.shape}")

        # Normalize per-image, avoid division by zero
        amin = arr.min()
        amax = arr.max()
        denom = (amax - amin)
        if denom <= 0:
            # if flat image, just zero it
            arr = np.zeros_like(arr, dtype=np.float32)
        else:
            arr = (arr - amin) / (denom + EPS)

        # Return torch tensor: shape (C, H, W)
        return torch.from_numpy(arr)