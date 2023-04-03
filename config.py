from collections import namedtuple


DatasetConfig = namedtuple('DatasetConfig', [
    'image_suffix',
    'label_suffix',
    'len_y',
    'len_x'
])


DATASET_CONFIGS = {
    'CHASE_DB1': DatasetConfig('.jpg', '_1stHO.png', 1024, 1024),
    'Artery': DatasetConfig('.jpg', '.bmp', 400, 400),
}