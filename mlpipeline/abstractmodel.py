from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass