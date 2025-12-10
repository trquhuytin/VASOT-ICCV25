# [Joint Self-Supervised Video Alignment and Action Segmentation (ICCV 2025)](https://arxiv.org/abs/2503.16832)

This repository contains our VASOT model for action segmentation only.

If you use the code, please cite our paper:
```
@inproceedings{ali2025joint,
  title={Joint Self-Supervised Video Alignment and Action Segmentation},
  author={Ali, Ali Shah and Mahmood, Syed Ahmed and Saeed, Mubin and Konin, Andrey and Zia, M Zeeshan and Tran, Quoc-Huy},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2025}
}
```

For our recent works, please check out our research page (https://retrocausal.ai/research/).

# Using the Code

## Run Unsupervised Learning Experiments

Install the dependencies into a conda environment using environment.yml  
Setup datasets as mentioned below and run the train/eval scripts.

## Datasets

Our datasets are comprised of per video frame features (pre-extracted), frame-wise labels and a mapping file which maps class IDs to action class names. Download instructions and folder structure are described in this section.

Breakfast, YTI, 50 Salads: [click here](https://github.com/Annusha/unsup_temp_embed/blob/master/HOWTO_master.md) to find links to download the datasets.
Desktop assembly: [click here](https://drive.google.com/drive/folders/1m7ljnnnd5kJ_Hi4Ir-sdNZRDFSqT1sHd) to download the dataset.

The data directory should at the minimum have the following structure.

```
data                 # root path for all datasets
├─ dataset_name/                # root path for single dataset
│  ├─ features/          # pre-extracted visual frame features
│  │  ├─ fname1.npy      # can also be txt
│  │  ├─ fname2.npy      # can also be txt
│  │  ├─ ...      
|  ├─ groundTruth/       # frame-wise labels
│  │  ├─ fname1 
│  │  ├─ fname2
│  │  ├─ ...      
|  ├─ mapping/       # frame-wise labels
│  │  ├─ mapping.txt # class-to-action ID mapping
```

`dataset_name` can be one of `Breakfast  desktop_assembly  FS  YTI`. It should be easy to set up new datasets as long as the folder structure is setup correctly.

## Dependencies

`numpy scipy scikit-learn matplotlib torch pytorch-lightning wandb`

## Run train/eval pipeline

We provide bash scripts and python commands to run the unsupervised learning experiments described in the paper. All hyperparameters are set in the subsequent scripts/commands according to the paper and should be consistent with the reported results.

### Breakfast

Run `bash run_bf.sh`. This runs training code for each activity class separately.

### YouTube Instructions

Run `bash run_yti.sh`. This runs training code for each activity class separately.

### 50 Salads, Desktop Assembly

```
python3 train.py -d FSeval -ac all -c 12 -ne 30 --seed 0 --group main_results --rho 0.15 -lat 0.11 -vf 5 -lr 1e-3 -wd 1e-4 -ua
python3 train.py -d FS -ac all -c 19 -ne 30 -g 0 --seed 0 --group main_results --rho 0.15 -lat 0.15 -vf 5 -lr 1e-3 -wd 1e-4 -ua
python3 train.py -d desktop_assembly -ac all -c 22 -ne 30 --seed 0 --group main_results --rho 0.25 -lat 0.16 -vf 5 -lr 1e-3 -wd 1e-4 -r 0.02 -ls 512 128 40 -ua
```
