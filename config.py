from collections import namedtuple


DatasetConfig = namedtuple('DatasetConfig', [
    'image_suffix',
    'label_suffix',
    'mask_suffix',
    'len_y',
    'len_x',
    'pixel_mean'
])


DATASET_CONFIGS = {
    'CHASE_DB1': DatasetConfig('.jpg', '_1stHO.png', '', 1024, 1024, [113.953, 39.807, 6.880]),
    'Artery': DatasetConfig('.bmp', '.tif', '', 400, 400, [127, 127, 127]),
    'HRF': DatasetConfig('.bmp', '.tif', '', 768, 768, [164.420, 51.826, 27.130])
}




##### Feature normalization #####

USE_BRN = True


##### Training (general) #####

MODEL_SAVE_PATH = 'train'
DISPLAY = 10
TEST_ITERS = 500
SNAPSHOT_ITERS = 500
WEIGHT_DECAY_RATE = 0.0005
MOMENTUM = 0.9
BATCH_SIZE = 2 # for CNN
GRAPH_BATCH_SIZE = 1 # for VGN

##### Training (augmentation) #####

# horizontal flipping
USE_LR_FLIPPED = True

# vertical flipping
USE_UD_FLIPPED = False

# rotation
USE_ROTATION = False
ROTATION_MAX_ANGLE = 45

# scaling
USE_SCALING = False
SCALING_RANGE = [1., 1.25]

# cropping
USE_CROPPING = False
CROPPING_MAX_MARGIN = 0.05 # in ratio

# brightness adjustment
USE_BRIGHTNESS_ADJUSTMENT = True
BRIGHTNESS_ADJUSTMENT_MAX_DELTA = 0.2

# contrast adjustment
USE_CONTRAST_ADJUSTMENT = True
CONTRAST_ADJUSTMENT_LOWER_FACTOR = 0.5
CONTRAST_ADJUSTMENT_UPPER_FACTOR = 1.5

##### Misc. #####

EPSILON = 1e-03


