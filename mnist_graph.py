from lib.datasets import MNIST as Data
from lib.model import Model as BaseModel, generate_placeholders, train
from lib.segmentation import extract_features_fixed
# from lib.segmentation import slic_fixed
from lib.segmentation import quickshift_fixed
from lib.pipeline import preprocess_pipeline_fixed
from lib.layer import EmbeddedGCNN as Conv, MaxPool, AveragePool, FC

# SLIC_FEATURES = [4, 5, 6, 7, 8, 18, 20, 21, 22]
QUICKSHIFT_FEATURES = [4, 6, 7, 8, 24, 28, 29, 31, 37]

DATA_DIR = 'data/mnist'

# PREPROCESS_FIRST = 'data/mnist/slic'
PREPROCESS_FIRST = 'data/mnist/quickshift'

LEVELS = 4
CONNECTIVITY = 8
SCALE_INVARIANCE = False
STDDEV = 1

LEARNING_RATE = 0.001
TRAIN_DIR = None
# LOG_DIR = 'data/summaries/mnist_slic_graph'
LOG_DIR = 'data/summaries/mnist_quickshift_graph'

AUGMENT_TRAIN_EXAMPLES = False
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

preprocess_algorithm = preprocess_pipeline_fixed(
    segmentation_algorithm, feature_extraction_algorithm, LEVELS, CONNECTIVITY,
    SCALE_INVARIANCE, STDDEV)


class Model(BaseModel):
    def _build(self):
        conv_1_1 = Conv(
            NUM_FEATURES,
            64,
            adjs_dist=self.placeholders['adj_dist_1'],
            adjs_rad=self.placeholders['adj_rad_1'],
            logging=self.logging)
        conv_1_2 = Conv(
            64,
            64,
            adjs_dist=self.placeholders['adj_dist_1'],
            adjs_rad=self.placeholders['adj_rad_1'],
            logging=self.logging)
        max_pool_1 = MaxPool(size=4)
        conv_2_1 = Conv(
            64,
            128,
            adjs_dist=self.placeholders['adj_dist_3'],
            adjs_rad=self.placeholders['adj_rad_3'],
            logging=self.logging)
        conv_2_2 = Conv(
            128,
            128,
            adjs_dist=self.placeholders['adj_dist_3'],
            adjs_rad=self.placeholders['adj_rad_3'],
            logging=self.logging)
        max_pool_2 = MaxPool(size=4)
        average_pool = AveragePool()
        fc_1 = FC(
            128,
            data.num_classes,
            act=lambda x: x,
            bias=False,
            dropout=self.placeholders['dropout'],
            logging=self.logging)

        self.layers = [
            conv_1_1, conv_1_2, max_pool_1, conv_2_1, conv_2_2, max_pool_2,
            average_pool, fc_1
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
