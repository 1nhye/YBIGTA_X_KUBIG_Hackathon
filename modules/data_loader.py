from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load_data(self):
        data = load_dataset(self.dataset_name, split="train")
        return data
