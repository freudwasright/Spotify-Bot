import os
import cv2
import torch
import numpy as np


def get_labels(df, column_name: str) -> torch.Tensor:
    """
    This function extracts the labels from a given pandas dataframe based on a column name of the categories. It also
    converts string-s to id-s, so the categories will be numerical.
    :param df: Pandas Dataframe object. The pre-loaded dataframe.
    :param column_name: String, dataframe attribute that contains the categories
    :return: torch.tensor object with a shape of (dataset_length, 1).
    """
    labels = df[column_name]
    labels_to_id = labels.apply(list(labels.unique()).index)
    tensor_labels = torch.tensor(labels_to_id)
    return tensor_labels


def get_image_files(root_dir: str) -> list:
    """
    The get_image_files function returns a list containing every file location that ends with a .png and is in the given
    root directory.
    :param root_dir: String, a file path.
    :return: List of file paths.
    """
    image_file_locations = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".png"):
                image_file_locations.append(filepath)
    return image_file_locations


def load_data(image_file_locations: list, transform=None) -> torch.Tensor:
    """
    This function loads the images based on the image file paths and then preprocesses the images if a transform object
    is given, if not the shapes will be transformed accordingly form (HxWxC) to (CxHxW).
    :param image_file_locations: list, list of file paths that point to the image file locations
    :param transform: torchvision.transform object, that preprocesses the images.
    :return: (DatasetSize x Channel x Height x Width) shaped torch.Tensor object.
    """
    #  [batch_size, channels, height, width].
    images = []
    for image_file in image_file_locations:
        current_img = cv2.imread(image_file).astype('uint8')
        if transform is not None:
            current_img = transform(current_img)
        else:
            current_img = torch.from_numpy(current_img).float() // 255.0
            current_img = torch.einsum('ijk->kij', current_img)  # reshape image
        images.append(current_img)
    data = torch.stack(images, dim=0)
    return data
