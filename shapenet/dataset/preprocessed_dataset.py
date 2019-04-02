import numpy as np

class DataSet:
    def __init__(self, path):
        ds = np.load(path)
        self.data = ds['data']
        self.labels = ds['labels']
        self.next_sample = 0

    def train_set_size(self):
        return self.data.shape[0]

    def next_batch(self, batch_size):
        d = self.data[self.next_sample:(self.next_sample + batch_size), :]
        l = self.labels[self.next_sample:(self.next_sample + batch_size), :]
        self.next_sample = self.next_sample + batch_size
        self.next_sample = self.next_sample if self.next_sample < self.train_set_size()
                                else 0
        return d, l
    
