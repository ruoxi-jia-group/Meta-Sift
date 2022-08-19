# Meta_sift

![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.10.1](https://img.shields.io/badge/pytorch-1.10.1-DodgerBlue.svg?style=plastic)
![CUDA 11.0](https://img.shields.io/badge/cuda-11.0-DodgerBlue.svg?style=plastic)



# Features
- Quickly sift out an clean subsets (about 8 mins in CIFAR-10)
- Applicable in most existing poison situations
- Applicable to most existing datasets
- Compatible with existing defense algorithms

# Requirements
+ Python >= 3.6
+ PyTorch >= 1.10.1
+ TorchVisison >= 0.11.2
+ Imageio >= 2.9.0

# Usage & HOW-TO
<p align="justify">Use the trojan_backdoor_detect_gtsrb.ipynb
 notebook for a quick start of our Meta-sift method. The default setting running on the GTSRB dataset and attack method is BadNets.</p>

There are a several of optional arguments in the ```args```:

- ```corruption_type```: The poison method
- ```corruption_ratio``` : The poison rates of the poison method.
- ```tar_lab``` : The label of the attack (if not a global poison)
- ```repeat_rounds``` : The number of sifters to use when selecting clean subsets .
- ```warmup_round``` : The number of epochs for warm-up before trainging sifters.
