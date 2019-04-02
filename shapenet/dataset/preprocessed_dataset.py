import numpy as np

class DataSet:
    def __init__(self, path):
        ds = np.load(path)
        self.data = ds['data']
        self.data = self.data.reshape(self.data.shape[0], 1, *self.data.shape[1:])
        self.labels = ds['labels']
        self.next_sample = 0

    def train_set_size(self):
        return self.data.shape[0]

    def next_batch(self, batch_size):
        n = self.next_sample + batch_size
        n = n if n < self.train_set_size() else self.train_set_size()
        d = self.data[self.next_sample:n, :]
        l = self.labels[self.next_sample:n, :]
        n = n if n < self.train_set_size() else 0
        self.next_sample = n
        return d, l

