from collections import namedtuple


DatasetConfig = namedtuple('DatasetConfig', [
    'image_suffix',
    'label_suffix',
    'mask_suffix',
    'len_y',
    'len_x',
    'pixel_mean',
    'gt_values'
])

Artery_vessel_values = {
    'blue': (29, 150),
    'red': (76, 150),
}

DATASET_CONFIGS = {
    'Artery': DatasetConfig('.bmp', '_all.bmp', '', 400, 400, [164.420, 51.826, 27.130], None),  # last value is the label values assigned to gt
    'HRF': DatasetConfig('.bmp', '.tif', '', 768, 768, [164.420, 51.826, 27.130], None)
}



##### Training (general) #####
DISPLAY = 10
TEST_ITERS = 100
SNAPSHOT_ITERS = 500


WEIGHT_DECAY_RATE = 0.0005
BATCH_SIZE = 8 # for CNN
GRAPH_BATCH_SIZE = 4 # for VGN

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


