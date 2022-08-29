# Meta-Sift

![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.10.1](https://img.shields.io/badge/pytorch-1.10.1-DodgerBlue.svg?style=plastic)
![CUDA 11.0](https://img.shields.io/badge/cuda-11.0-DodgerBlue.svg?style=plastic)

Most existing defenses against data poisoning assume access to a set of clean data (referred to as the base set hereinafter). While this assumption has been taken for granted, given the fast-growing research on tealthy data poisoning techniques, an important question arises: can the defender really identify a clean subset within a contaminated dataset to support the defenses? We accomplish extensive experiments to give a negative answer to this question. 
<br>

![Narcissus-Caravaggio](./visual.png)
Human inspection results regarding data poisoning attacks. The labels and images marked in red depict potential manipulations under that attack category, and the green represents that the attribute remains intact. Among the three different categories of attacks, we report the error rate of misclassifying clean samples into poisoned ones (FPR) or poisoned ones into clean samples (FNR). The result reveals even humans can't identify all poisoned samples.


In addition to uncovering the challenge of identifying a clean base set with high precision, we take a step further and propose META-SIFT to resolve the challenge. Our evaluation shows that META-SIFT can robustly sift out a clean base set with 100% precision under a wide range of poisoning attacks. The selected base set is large enough to give rise to successful defense when plugged into the existing defense techniques.

# Features
- Quickly sift out an clean subsets (about 8 mins on the CIFAR-10)
- Applicable in most existing poisoning situations
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
