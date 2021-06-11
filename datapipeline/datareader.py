from datapipeline import utils
from torch.utils.data import Dataset
import pandas as pd


class MusicDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, transform=None):
        """
        Constructor for MusicDataset sub-classed class. initializes self.samples, self.transform, self.labels, and
        self.data.
        :param csv_path: Path of the csv folder which contains the features and file names for the songs.
        :param root_dir: Root directory of the mel spectrogram images.
        :param transform: PyTorch transform class, that preprocessing the images.
        """
        # data loading
        meta_data = pd.read_csv(csv_path)
        song_file_names = utils.get_image_files(root_dir)

        self.n_samples = meta_data.shape[0]
        self.transform = transform
        self.labels = utils.get_labels(meta_data, column_name='label')
        self.data = utils.load_data(song_file_names, self.transform)

    def __getitem__(self, index):
        """
        This class method inherits from the Dataset class and creates and handles the proper indexing of elements.
        :param index: Data index, integer.
        :return: The correspondent data.
        """
        # dataset indexing
        return self.data[index], self.labels[index]

    def __len__(self):
        """
        This class method inherits from the Dataset class and returns the length of the dataset.
        :return: Integer, length of the dataset.
        """
        # get length of dataset
        return self.n_samples
