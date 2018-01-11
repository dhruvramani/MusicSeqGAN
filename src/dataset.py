import os
import numpy

class Dataset(object):
    def __init__(self, num_batches, batch_size, steps, embedding_dim):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.steps = steps
        self.embedding_dim = embedding_dim

    def string_vectorizer(self, strng, alphabet=string.ascii_lowercase):
        # SOURCE : https://stackoverflow.com/questions/43618245/how-to-one-hot-encode-sentences-at-the-character-level
        vector = [[0 if char != letter else 1 for char in alphabet] for letter in strng]
        return vector

    def get_data(self, datatype):
        filepath = "./dataset/{}/".format(datatype)
        n_steps = list()
        for i in batch_size:
            with open("{}{}.abc".format(filepath, i), "r") as f:
                content = f.read()
                n_steps.append(string_vectorizer(content))
        n_steps = np.asarray(n_steps)
        return n_steps

    def get_batch(self, datatype):
        for i in num_batches:
            yield get_data(datatype)