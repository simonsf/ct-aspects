from easydict import EasyDict as edict
from modules.normalizer import FixedWindowNormalizer
import numpy as np

__C = edict()
cfg = __C


##################################
# general parameters
##################################

__C.general = {}

# image-segmentation pair list
# 1) single-modality image training, use txt annotation file
# 2) multi-modality image training, use csv annotation file
__C.general.im_list = ".././train.txt"
__C.general.tag_list = ".././tags.csv"
#__C.general.cbf_file = ".././cbf.csv"

# the output of training models and logs
__C.general.save_dir = 'model'
# continue training from certain epoch, -1 to train from scratch
__C.general.resume_epoch = -1
# the number of GPUs used in training
__C.general.num_gpus = 2

# random seed used in training (debugging purpose)
__C.general.seed = 0

# ASPECTS region id
__C.general.region = None

##################################
# data set parameters
##################################

__C.dataset = {}

# the number of classes
__C.dataset.num_classes = 2

# whether to use ASPECTS-region masks
__C.dataset.mask = True

# the resolution on which segmentation is performed
__C.dataset.spacing = [0.5, 0.5]

# the sampling crop size, e.g., determine the context information
__C.dataset.crop_size = [7, 130, 110]

# the default padding value list
__C.dataset.default_values = [-1]

# sampling method:
# 1) GLOBAL: sampling crops randomly in the entire image domain
# 2) MASK: sampling crops randomly within segmentation mask
__C.dataset.sampling_method = 'GLOBAL'

# translation augmentation (unit: voxel)
__C.dataset.random_translation = [5, 5]

# linear interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'LINEAR'

# crop intensity normalizers (to [-1,1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use minimum and maximum intensity of crop to normalize intensity
__C.dataset.crop_normalizers = [FixedWindowNormalizer(30, 60)]

####################################
# training loss
####################################

__C.loss = {}

# the name of loss function to use
# Focal: Focal loss, supports binary-class and multi-class segmentation
# Dice: Dice Similarity Loss, supports binary-class and multi-class segmentation
__C.loss.name = 'CE'

# the weight for each class including background class
# weights will be normalized
__C.loss.obj_weight = [1, 1]

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2


#####################################
# net
#####################################

__C.net = {}

# the network name
__C.net.name = 'densenet'
# whether to use NWU as additional fc feature
__C.net.nwu = False


######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
__C.train.epochs = 401

# the number of samples in a batch
__C.train.batchsize = 30

# the number of threads for IO
__C.train.num_threads = 1

# the learning rate
__C.train.lr = 6e-5

# the beta in Adam optimizer
__C.train.betas = (0.9, 0.999)

# the number of batches to update loss curve
__C.train.plot_snapshot = 3

# the number of batches to save model
__C.train.save_epochs = 20


########################################
# debug parameters
########################################

__C.debug = {}

# whether to save input crops
__C.debug.save_inputs = True

