# ct-aspects

# DA-Net
### Introduction

This repository is for the proposed DA-Net used for AI-assisted ASPECTS in NCCT.

[Cao Z, Xu J, Song B, Chen L, Sun T, He Y, Wei Y, Niu G, Zhang Y, Feng Q, Ding Z. Deep learning derived automated ASPECTS on non‐contrast CT scans of acute ischemic stroke patients. *Human Brain Mapping; 2022 Mar 31*.](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.25845)


![此处填入框图地址](https://github.com/simonsf/ct-aspects/tree/main/img.png)

-------------------------------

### Content

-  [data:](https://github.com/simonsf/ct-aspects/tree/main/data) this folder contains five subjects along with whole processdure.

-  [DenseNet.py:](https://github.com/simonsf/ct-aspects/blob/main/DenseNet.py) contains the structure of our model.

-  [WeightedMSE.py, ](https://github.com/simonsf/ct-aspects/blob/main/WeightedMSE.py) [AvgLoss.py, ](https://github.com/simonsf/ct-aspects/blob/main/AvgLoss.py)[RankLoss.py:](https://github.com/simonsf/ct-aspects/blob/main/RankLoss.py) implements difference loss functions used in our training processing.

-  [train_cortical.py:](https://github.com/simonsf/ct-aspects/blob/main/train_cortical.py) contains the functions to train the network, and also needs to be fitness to your task and data with some changes.

-  [test.py:](https://github.com/simonsf/ct-aspects/blob/main/test.py)contain the functions to test the trained model, and also needs to be fitness to your task and data with some changes.

- [config.py:](https://github.com/simonsf/ct-aspects/blob/main/test.py) contain configurations during the training.  

----------------------

### Requirement

This repository is based on PyTorch 1.1.0, developed in Ubuntu 16.04 environment. 


-  cuda(required by pytorch), cudnn, numpy, scipy, sklearn, tqdm, pillow, matplotlib, ipython, ...

-----------------------

Due to potential commercial issue, the trained model on large dataset was not included.
