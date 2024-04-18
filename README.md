# Divide and Adapt: Active Domain Adaptation via Customized Learning (CVPR 2023 Highlight Paper)
Pytorch official implementation for our CVPR-2023 highlight paper "Divide and Adapt: Active Domain Adaptation via Customized Learning". More details of this work can be found in our paper: [[Arxiv]](https://arxiv.org/abs/2307.11618) or [[OpenAccess]](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Divide_and_Adapt_Active_Domain_Adaptation_via_Customized_Learning_CVPR_2023_paper.html).

Our code is based on [CLUE](https://github.com/virajprabhu/CLUE) implementation.


## Abstract

Active domain adaptation (ADA) aims to improve the model adaptation performance by incorporating the active learning (AL) techniques to label a maximally-informative subset of target samples. Conventional AL methods do not consider the existence of domain shift, and hence, fail to identify the truly valuable samples in the context of domain adaptation. To accommodate active learning and domain adaption, the two naturally different tasks, in a collaborative framework, we advocate that a customized learning strategy for the target data is the key to the success of ADA solutions. We present Divide-and-Adapt (DiaNA), a new ADA framework that partitions the target instances into four categories with stratified transferable properties. With a novel data subdivision protocol based on uncertainty and domainness, DiaNA can accurately recognize the most gainful samples. While sending the informative instances for annotation, DiaNA employs tailored learning strategies for the remaining categories. Furthermore, we propose an informativeness score that unifies the data partitioning criteria. This enables the use of a Gaussian mixture model (GMM) to automatically sample unlabeled data into the proposed four categories. Thanks to the "divide-and-adapt" spirit, DiaNA can handle data with large variations of domain gap. In addition, we show that DiaNA can generalize to different domain adaptation settings, such as unsupervised domain adaptation (UDA), semi-supervised domain adaptation (SSDA), source-free domain adaptation (SFDA), etc.


## Dependencies and Datasets
1. Create an anaconda environment with [Python 3.6](https://www.python.org/downloads/release/python-365/) and activate: 
```
conda create -n diana python=3.6.8
conda activate diana
```
2. Install dependencies: 
```
pip install -r requirements.txt
``` 



## Download data
For DomainNet, follow the following steps:
1. Download the original dataset for the domains of interest from [this link](http://ai.bu.edu/M3SDA/) – eg. Clipart and Sketch.
2. Run: 
```
python preprocess_domainnet.py --input_dir data/domainNet/ \
                               --domains 'clipart,sketch' \
                               --output_dir 'data/post_domainNet/'
```


## Pretrained checkpoints
At round 0, active adaptation begins from a model trained on the source domain, or from a model first trained on source and then adapted to the target via unsupervised domain adaptation. Skip the this step if you want to train from scratch. Otherwise, download the checkpoints pretrained on each source domain at [this link](https://drive.google.com/file/d/18aYIgRTU_ERfcTLNQC_b4tYzimgFC1xe/view?usp=sharing) and unzip them to ```checkpoints/source/domainnet``` folder. 

## Train Active Domain Adaptation model
We include hyperparameter configurations to reproduce paper numbers on DomainNet as configurations inside the ```config``` folder. For instance, to reproduce DomainNet (Clipart->Sketch) results with DiaNA. Run: 
```
python train.py --cfg_file config/domainnet/clipart2sketch.yml -a GMM -d self_ft
```

To reproduce results with CLUE+MME, run:
```
python train.py --cfg_file config/domainnet/clipart2sketch.yml -a CLUE -d mme
```

For other baseline methods, run:
```
python train.py --cfg_file config/domainnet/clipart2sketch.yml -a uniform
```
All the supported baselines currently:
* uniform(random)
* [Alpha](https://arxiv.org/abs/2203.07034)
* [BADGE](https://arxiv.org/abs/1906.03671)
* [Entropy](https://ieeexplore.ieee.org/document/6889457)



<!-- ## Combination with DA methods -->
 

## Citation and Star
Please cite the following paper and star this project if you use this repository in your research. Thank you!
```
@inproceedings{huang2023divide,
  title={Divide and adapt: Active domain adaptation via customized learning},
  author={Huang, Duojun and Li, Jichang and Chen, Weikai and Huang, Junshi and Chai, Zhenhua and Li, Guanbin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7651--7660},
  year={2023}
}
```


