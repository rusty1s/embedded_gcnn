from lib.datasets import MNIST as Data
from lib.model import Model as BaseModel, generate_placeholders, train
from lib.segmentation import extract_features_fixed
# from lib.segmentation import slic_fixed
from lib.segmentation import quickshift_fixed
from lib.pipeline import preprocess_pipeline_fixed
from lib.layer import EmbeddedGCNN as Conv, MaxPool, AveragePool, FC

# SLIC_FEATURES = [2, 3, 4, 5, 13, 13, 18, 29, 34]
QUICKSHIFT_FEATURES = [17, 24, 25, 26, 28, 29, 31, 33, 36]

DATA_DIR = 'data/mnist'
PREPROCESS_FIRST = False

LEVELS = 4
SCALE_INVARIANCE = False
STDDEV = 1

LEARNING_RATE = 0.001
TRAIN_DIR = None
# LOG_DIR = 'data/summaries/mnist_slic_embedded'
LOG_DIR = 'data/summaries/mnist_quickshift_embedded'

DROPOUT = 0.5
BATCH_SIZE = 64
MAX_STEPS = 20000
DISPLAY_STEP = 10
# FORM_FEATURES = SLIC_FEATURES
FORM_FEATURES = QUICKSHIFT_FEATURES
NUM_FEATURES = len(FORM_FEATURES) + 1

data = Data(DATA_DIR)

# segmentation_algorithm = slic_fixed(
#     num_segments=100, compactness=5, max_iterations=10, sigma=0)
segmentation_algorithm = quickshift_fixed(
    ratio=1, kernel_size=2, max_dist=2, sigma=0)

feature_extraction_algorithm = extract_features_fixed(FORM_FEATURES)

filter_algorithm = None

preprocess_algorithm = preprocess_pipeline_fixed(
    segmentation_algorithm, feature_extraction_algorithm, LEVELS,
    filter_algorithm, SCALE_INVARIANCE, STDDEV)


class Model(BaseModel):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.build()

    def _build(self):
        conv_1 = Conv(
            NUM_FEATURES,
            32,
            adjs_dist=self.placeholders['adj_dist_1'],
            adjs_rad=self.placeholders['adj_rad_1'],
            logging=self.logging)
        max_pool_1 = MaxPool(size=2)
        conv_2 = Conv(
            32,
            64,
            adjs_dist=self.placeholders['adj_dist_2'],
            adjs_rad=self.placeholders['adj_rad_2'],
            logging=self.logging)
        max_pool_2 = MaxPool(size=2)
        conv_3 = Conv(
            64,
            128,
            adjs_dist=self.placeholders['adj_dist_3'],
            adjs_rad=self.placeholders['adj_rad_3'],
            logging=self.logging)
        max_pool_3 = MaxPool(size=2)
        conv_4 = Conv(
            128,
            256,
            adjs_dist=self.placeholders['adj_dist_4'],
            adjs_rad=self.placeholders['adj_rad_4'],
            logging=self.logging)
        max_pool_4 = MaxPool(size=2)
        average_pool = AveragePool()
        fc_1 = FC(256, 128, weight_decay=0.004, logging=self.logging)
        fc_2 = FC(128,
                  data.num_classes,
                  dropout=self.placeholders['dropout'],
                  act=lambda x: x,
                  logging=self.logging)

        self.layers = [
            conv_1, max_pool_1, conv_2, max_pool_2, conv_3, max_pool_3, conv_4,
            max_pool_4, average_pool, fc_1, fc_2
        ]


placeholders = generate_placeholders(BATCH_SIZE, LEVELS, NUM_FEATURES,
                                     data.num_classes, DROPOUT)

model = Model(
    placeholders=placeholders,
    learning_rate=LEARNING_RATE,
    train_dir=TRAIN_DIR,
    log_dir=LOG_DIR)

train(model, data, preprocess_algorithm, BATCH_SIZE, DROPOUT, MAX_STEPS,
      PREPROCESS_FIRST, DISPLAY_STEP)
