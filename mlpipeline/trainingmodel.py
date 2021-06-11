from abstractmodel import AbstractModel


class MLPipeline(AbstractModel):
    def __init__(self, model):
        self.base_model = model
