from os.path import join

from yacs.config import CfgNode

from config.const import PROJECT_ROOT


_C = CfgNode()
_C.SEED = 42


# train
_C.TRAIN = CfgNode()
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.CRITERION_TYPE = 'cross_entropy'
_C.TRAIN.EPOCH_NUM = 30
_C.TRAIN.LR = 0.01
_C.TRAIN.MODEL_TYPE = 'simple_cnn'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER_TYPE = 'sgd'
_C.TRAIN.SCHEDULER_TYPE = 'step_lr'
_C.TRAIN.STEP_SIZE = 5

_C.TEST = CfgNode()


_C.DATA = CfgNode()
_C.DATA.CACHE_DIR = join(PROJECT_ROOT, '.data')
_C.DATA.CLASSES = (
        'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        )
_C.DATA.DATASET_TYPE = 'cifer10'
_C.DATA.INPUT_DIM = 3
_C.DATA.TRANSFORM_LIST = ['to_tensor', 'normalize']
_C.DATA.VAL_SIZE = 0.25
