from lib.datasets import MNIST as Data
from lib.model import Model as BaseModel, generate_placeholders, train
from lib.segmentation import extract_features_fixed
# from lib.segmentation import slic_fixed
from lib.segmentation import quickshift_fixed
from lib.pipeline import preprocess_pipeline_fixed
from lib.layer import EmbeddedGCNN as Conv, MaxPool, AveragePool, FC

# SLIC_FEATURES = [0, 2, 5, 7, 8, 9, 19, 21, 22]
QUICKSHIFT_FEATURES = [2, 3, 4, 5, 7, 8, 19, 21, 22]

DATA_DIR = 'data/cifar_10'

# PREPROCESS_FIRST = 'data/cifar_10/slic'
PREPROCESS_FIRST = 'data/cifar_10/quickshift'

LEVELS = 4
SCALE_INVARIANCE = False
STDDEV = 1

LEARNING_RATE = 0.001
TRAIN_DIR = None
# LOG_DIR = 'data/summaries/cifar_slic_embedded'
LOG_DIR = 'data/summaries/cifar_quickshift_embedded'

AUGMENT_TRAIN_EXAMPLES = False
DROPOUT = 0.5
BATCH_SIZE = 64
MAX_STEPS = 200000
DISPLAY_STEP = 10
# FORM_FEATURES = SLIC_FEATURES
FORM_FEATURES = QUICKSHIFT_FEATURES
NUM_FEATURES = len(FORM_FEATURES) + 3

data = Data(DATA_DIR)

# segmentation_algorithm = slic_fixed(
#     num_segments=200, compactness=5, max_iterations=10, sigma=0)
segmentation_algorithm = quickshift_fixed(
    ratio=1, kernel_size=1, max_dist=5, sigma=0)

feature_extraction_algorithm = extract_features_fixed(FORM_FEATURES)

preprocess_algorithm = preprocess_pipeline_fixed(
    segmentation_algorithm, feature_extraction_algorithm, LEVELS,
    SCALE_INVARIANCE, STDDEV)


class Model(BaseModel):
    def _build(self):
        conv_1_1 = Conv(
            NUM_FEATURES,
            32,
            adjs_dist=self.placeholders['adj_dist_1'],
            adjs_rad=self.placeholders['adj_rad_1'],
            logging=self.logging)
        conv_1_2 = Conv(
            32,
            32,
            adjs_dist=self.placeholders['adj_dist_1'],
            adjs_rad=self.placeholders['adj_rad_1'],
            logging=self.logging)
        max_pool_1 = MaxPool(size=2)
        conv_2_1 = Conv(
            32,
            64,
            adjs_dist=self.placeholders['adj_dist_2'],
            adjs_rad=self.placeholders['adj_rad_2'],
            logging=self.logging)
        conv_2_2 = Conv(
            64,
            64,
            adjs_dist=self.placeholders['adj_dist_2'],
            adjs_rad=self.placeholders['adj_rad_2'],
            logging=self.logging)
        max_pool_2 = MaxPool(size=2)
        conv_3_1 = Conv(
            64,
            128,
            adjs_dist=self.placeholders['adj_dist_3'],
            adjs_rad=self.placeholders['adj_rad_3'],
            logging=self.logging)
        conv_3_2 = Conv(
            128,
            128,
            adjs_dist=self.placeholders['adj_dist_3'],
            adjs_rad=self.placeholders['adj_rad_3'],
            logging=self.logging)
        max_pool_3 = MaxPool(size=2)
        conv_4_1 = Conv(
            128,
            256,
            adjs_dist=self.placeholders['adj_dist_4'],
            adjs_rad=self.placeholders['adj_rad_4'],
            logging=self.logging)
        conv_4_2 = Conv(
            256,
            256,
            adjs_dist=self.placeholders['adj_dist_4'],
            adjs_rad=self.placeholders['adj_rad_4'],
            logging=self.logging)
        max_pool_4 = MaxPool(size=2)
        average_pool = AveragePool()
        fc_1 = FC(256, 384, weight_decay=0.004, logging=self.logging)
        fc_2 = FC(384, 192, weight_decay=0.004, logging=self.logging)
        fc_3 = FC(192,
                  data.num_classes,
                  act=lambda x: x,
                  bias=False,
                  dropout=self.placeholders['dropout'],
                  logging=self.logging)

        self.layers = [
            conv_1_1, conv_1_2, max_pool_1, conv_2_1, conv_2_2, max_pool_2,
            conv_3_1, conv_3_2, max_pool_3, conv_4_1, conv_4_2, max_pool_4,
            average_pool, fc_1, fc_2, fc_3
        ]


placeholders = generate_placeholders(BATCH_SIZE, LEVELS, NUM_FEATURES,
                                     data.num_classes)

model = Model(
    placeholders=placeholders,
    learning_rate=LEARNING_RATE,
    train_dir=TRAIN_DIR,
    log_dir=LOG_DIR)

train(model, data, preprocess_algorithm, BATCH_SIZE, DROPOUT,
      AUGMENT_TRAIN_EXAMPLES, MAX_STEPS, PREPROCESS_FIRST, DISPLAY_STEP)
