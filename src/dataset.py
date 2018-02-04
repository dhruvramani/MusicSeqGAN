import os
import numpy as np

class Dataset(object):
    def __init__(self, num_batches, batch_size, steps, embedding_dim):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.steps = steps
        self.embedding_dim = embedding_dim

    def string_vectorizer(self, strng, alphabet=[chr(i) for i in range(0, 256)]):
        # SOURCE : https://stackoverflow.com/questions/43618245/how-to-one-hot-encode-sentences-at-the-character-level
        vector = [[0 if char != letter else 1 for char in alphabet] for letter in strng]
        return vector

    def get_data(self, datatype):
        filepath = "./dataset/{}/".format(datatype)
        x, y = list(), list()
        for i in range(self.batch_size):
            x_path, y_path = "{}piano/{}.abc".format(filepath, i), "{}guitar/{}.abc".format(filepath, i)
            if(not os.path.isfile(x_path) or not os.path.isfile(y_path)):
                continue
            with open(x_path, "r") as f:
                content = f.read()
                x.append(self.string_vectorizer(content))
            
            with open(y_path, "r") as f:
                content = f.read()
                y.append(self.string_vectorizer(content))
        
        if(0 in [len(x), len(y)]):
            return None, None
        x, y = np.asarray(x), np.asarray(y)
        return x, y

    def get_batch(self, datatype):
        for i in range(self.num_batches):
            data = self.get_data(datatype)
            if(None not in data):
                yield data