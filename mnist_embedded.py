from lib.datasets import MNIST as Data
from lib.model import Model as BaseModel, generate_placeholders, train
from lib.segmentation import (slic_fixed, mnist_slic_feature_extraction,
                              NUM_MNIST_SLIC_FEATURES)
from lib.pipeline import preprocess_pipeline_fixed
from lib.layer import EmbeddedGCNN as Conv, MaxPool, AveragePool, FC

DATA_DIR = 'data/mnist'

LEVELS = 4
SCALE_INVARIANCE = False
STDDEV = 1

LEARNING_RATE = 0.1
TRAIN_DIR = None
LOG_DIR = None

DROPOUT = 0.5
BATCH_SIZE = 64
MAX_STEPS = 20000
DISPLAY_STEP = 10

NUM_FEATURES = NUM_MNIST_SLIC_FEATURES

data = Data(DATA_DIR)

segmentation_algorithm = slic_fixed(
    num_segments=100, compactness=2, max_iterations=10, sigma=0)

feature_extraction_algorithm = mnist_slic_feature_extraction

preprocess_algorithm = preprocess_pipeline_fixed(
    segmentation_algorithm, feature_extraction_algorithm, LEVELS,
    SCALE_INVARIANCE, STDDEV)


class Model(BaseModel):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.build()

    def _build(self):
        conv_1 = Conv(
            6,
            32,
            adjs_dist=self.placeholders['adj_dist_1'],
            adjs_rad=self.placeholders['adj_rad_1'])
        max_pool_1 = MaxPool(size=2)
        conv_2 = Conv(
            32,
            64,
            adjs_dist=self.placeholders['adj_dist_2'],
            adjs_rad=self.placeholders['adj_rad_2'])
        max_pool_2 = MaxPool(size=2)
        conv_3 = Conv(
            64,
            128,
            adjs_dist=self.placeholders['adj_dist_3'],
            adjs_rad=self.placeholders['adj_rad_3'])
        max_pool_3 = MaxPool(size=2)
        conv_4 = Conv(
            128,
            256,
            adjs_dist=self.placeholders['adj_dist_4'],
            adjs_rad=self.placeholders['adj_rad_4'])
        max_pool_4 = MaxPool(size=2)
        average_pool = AveragePool()
        fc_1 = FC(256, 128)
        fc_2 = FC(128,
                  data.num_labels,
                  dropout=self.placeholders['dropout'],
                  act=lambda x: x,
                  logging=self.logging)

        self.layers = [
            conv_1, max_pool_1, conv_2, max_pool_2, conv_3, max_pool_3, conv_4,
            max_pool_4, average_pool, fc_1, fc_2
        ]


placeholders = generate_placeholders(BATCH_SIZE, LEVELS, NUM_FEATURES,
                                     data.num_labels, DROPOUT)

model = Model(
    placeholders=placeholders,
    learning_rate=LEARNING_RATE,
    train_dir=TRAIN_DIR,
    log_dir=LOG_DIR)

train(model, data, preprocess_algorithm, BATCH_SIZE, DROPOUT, MAX_STEPS,
      DISPLAY_STEP)
