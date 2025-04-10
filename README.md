
# Label Dependency Aware Loss for Reliable Multi-Label Medical Image Classification

This is the official PyTorch training code for LDACE and CCL loss used for training on ChestMNIST, PTB-XL and RFMiD dataset. The code is trained using PyTorch version 2.1.2 with cuda enabled and torchvision version 0.16.2.

## Abstract
A key challenge in multi-label classification is to model the dependencies between the labels while ensuring proper calibration, as the assumption of label independence often results in inferior classification performance and poor calibration. However, most of the earlier works that modeled label dependencies have neglected the problem of ensuring calibrated results, which is crucial in safety-critical applications like medical image analysis. In this paper, we propose a novel training loss function, Label Dependency Aware Cross Entropy (LDACE), specifically designed for capturing pairwise label dependencies during the learning process. Additionally, we introduce an auxiliary loss, Canonical Calibration Loss (CCL), which when combined with LDACE ensures better calibration. We evaluate the effectiveness of the proposed loss function through experiments on three publicly available datasets -- ChestMNIST, PTB-XL and RFMiD -- by comparing it against the traditional multi-label losses using various deep learning models. The classification and calibration results demonstrate the superiority of the proposed loss function over the traditional loss functions in terms of Hamming loss, Area Under the Receiver Operating Characteristic Curve (AUC), Average Calibration Error (ACE) and Maximum Calibration Error (MCE). Furthermore, the experiments also show that the auxiliary loss not only improves the calibration performance but also retains the classification performance.

## Training
Before initiating the training or evaluation process, please ensure that the required datasets are downloaded and a corresponding data loader is appropriately designed for the multi-label dataset. You may consider modifying the `dataloader.py` file accordingly. The current implementation supports LDACE and CCL methods on the ChestMNIST dataset.

Please update the relevant configurations in the `main.py` file, specifically between lines **55–69**, to suit your dataset and training setup.

To begin training, execute the following command:
```
$> python main.py
```
## Evalutation
To evaluate a trained model, ensure that the configuration in `model_eval.py` (lines **14–29**) matches that of `main.py`. Additionally, specify the path to the model checkpoint in the configuration section.

Run the evaluation script using the following command:
```
$> python model_eval.py
```
## Using the code
The code included in this GitHub repo. can be used, free of charge, for research and educational purposes. Copying, redistribution, and any unauthorized commercial use are prohibited. Any researcher reporting results which use this code must cite this publication:
```
@inproceedings{pal2025label,
  title={Label Dependency Aware Loss for Reliable Multi-Label Medical Image Classification},
  author={Pal, Aditya Shankar and Panda, Arkapal and Garain, Utpal},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
Feedback on the code is welcome and can be mailed to *eddie.aditya14@gmail.com*.

