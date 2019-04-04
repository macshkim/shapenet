import numpy as np

class DataSet:
    def __init__(self, path):
        ds = np.load(path)
        self.data = ds['data'][0:1]
        self.data = self.data.reshape(self.data.shape[0], 1, *self.data.shape[1:])
        self.labels = ds['labels'][0:1]
        self.next_sample = 0

    def set_size(self):
        return self.data.shape[0]

    def next_batch(self, batch_size):
        n = self.next_sample + batch_size
        n = n if n < self.set_size() else self.set_size()
        d = self.data[self.next_sample:n, :]
        l = self.labels[self.next_sample:n, :]
        n = n if n < self.set_size() else 0
        self.next_sample = n
        return d, l

    def get_label(self, idx):
        return self.labels[idx]

