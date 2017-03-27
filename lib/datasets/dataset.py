import os


class DataSet(object):
    def __init__(self, data_dir=None):
        self.data_dir = data_dir

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def next_train_batch(self, batch_size):
        raise NotImplementedError

    def next_validation_batch(self, batch_size):
        raise NotImplementedError

    def next_test_batch(self, batch_size):
        raise NotImplementedError
