# Meta-Sift

![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.10.1](https://img.shields.io/badge/pytorch-1.10.1-DodgerBlue.svg?style=plastic)
![CUDA 11.0](https://img.shields.io/badge/cuda-11.0-DodgerBlue.svg?style=plastic)

Most existing defenses against data poisoning assume access to a set of clean data (referred to as the <b>base set</b>). While this assumption has been taken for granted, given the fast-growing research on stealthy data poisoning techniques, we find that <b>defenders with existing methods, including manual inspections, cannot identify a clean base set within a contaminated dataset</b>. 
<br>

![Humanexp](https://user-images.githubusercontent.com/77789132/187101117-15aaa0e5-1d6c-4282-9ec2-2aabfe25270e.jpg)
The above figure shows the human inspection results regarding data poisoning attacks. The labels and images marked in red depict potential manipulations under that attack category, and the green represents that the attribute remains intact. Among the three types of attacks, we report the error rate of misclassifying clean samples into poisoned ones (FPR) or poisoned ones into clean samples (FNR). The result reveals humans indeed can't identify all poisoned samples with high precision. In particular, manual inspection's performance in identifying Feature-Only (e.g., clean-label backdoor attacks) attacks is only marginally better than random selection.


With the above-identified challenge of obtaining a clean base set with high precision, we take a step further and propose META-SIFT to resolve the challenge. Our evaluation shows that META-SIFT can robustly sift out a clean base set (size 1000 or more) with 100% precision and zero variance under a wide range of poisoning attacks. 
<b>The selected base set is large enough to give rise to successful defense when plugged into the existing AI-security defense techniques</b> 
(e.g., robust training for <a href="https://github.com/xjtushujun/meta-weight-net">mitigating label-noise attacks</a>; <a href="https://github.com/AI-secure/Meta-Nerual-Trojan-Detection">trojan-net detections</a>, <a href="https://github.com/ruoxi-jia-group/I-BAU">backdoor removal defenses</a>, or <a href="https://github.com/ruoxi-jia-group/frequency-backdoor">backdoor sample detections</a>).

# Features
- Quickly sift out clean subsets (about 80 seconds on the CIFAR-10 with 5 GPUs)
- No need for pre-training any model
- Effective against most existing poisoning attack settings (evaluated on 16 existing label-flipping, backdoor, poisoning attacks)
- Applicable to most existing datasets (evaluated on CIFAR-10, GTSRB, PubFig, ImageNet)
- Can be adopted as n off-the-shelf toll and give rise to existing defense algorithms under settings where no clean base set access

# Requirements
+ Python >= 3.6
+ PyTorch >= 1.10.1
+ Torchvision >= 0.11.2
+ Imageio >= 2.9.0

# Usage & HOW-TO
<p align="justify">Use the trojan_backdoor_detect_gtsrb.ipynb
 notebook for a quick start of the Meta-Sift method (demonstrated on the GTSRB dataset). The default setting running on the GTSRB dataset and attack method is BadNets.</p>

There are a several of optional arguments in the ```args```:

- ```corruption_type```: The poison method
- ```corruption_ratio``` : The poison rates of the poison method.
- ```tar_lab``` : The target label of the attack (if not targeting at all the labels)
- ```repeat_rounds``` : The number of sifters to use when selecting clean subsets, default 5.
- ```warmup_round``` : The number of epochs for warm-up before training sifters, default 1.

# Overall Workfolw
![wholeworkflow](https://user-images.githubusercontent.com/77789132/187102168-1106e405-477f-4f63-86ae-b980b356a7a8.jpg)


The whole process of Meta-Sift consists of two stages: the <b>Training</b> Stage and the <b>Identification Stage</b>. Multiple(m) Sifters will be included during the <b>Identification Stage</b> to reduce the randomness resulting from SGD and randomized sample-dilution. As such, the <b>Training Stage</b> will be repeated m times with different random seeds to obtain m Sifters. In each Sifter, there are two different structures working as a pair: model θ and MW-Net ψ. In one iteration of the <b>Training Stage</b>, there are four steps: Virtual-update of θ; Gradient Sampling using the meta-gradient-sampler Γ; Meta-update of ψ; then the Actual-update of θ. After only one iteration,<b>Training Stage</b> will terminate. The trained Sifters will be adopted in the <b>Identification Stage</b> to assign weights to the diluted data from the dataset. Finally, Meta-Sift aggregates the results from multiple Sifters, and the clean samples will be sifted by inspecting the high-value end.


# Special thanks to...
[![Stargazers repo roster for @ruoxi-jia-group/Meta-Sift](https://reporoster.com/stars/ruoxi-jia-group/Meta-Sift)](https://github.com/ruoxi-jia-group/Meta-Sift/stargazers)
