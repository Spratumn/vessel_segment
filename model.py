import config as cfg
from module import DRIU, LargeDRIU



class VesselSegmCNN():
    def __init__(self, params):
        self.params = params
        if params.cnn_model == 'driu':
            self.build_driu()
        elif params.cnn_model == 'driu_large':
            self.build_driu_large()
        else:
            raise ValueError('Invalid cnn_model params!')
        print("Model built.")

    def build_driu(self):
        self.cnn_model = DRIU()

    def build_driu_large(self):
        self.cnn_model = LargeDRIU()


