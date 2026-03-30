# CLAVAS: Contrastive Learning – Adversarial Vulnerability And Susceptibility

## Acknowledgement
- This project originated from the Artificial Intelligence-based Image Recognition course in the Graduate School of Gachon University.

## Abstract
- Recently, **self-supervised learning (SSL)** and **contrastive learning (CL)** have been widely used to learn a model’s representation layer from unlabeled data.
- However, contrastive learning treats two different images as a **negative pair** even when they belong to the same class.
- This learning strategy can increase the distance between samples within the same class, unnecessarily enlarging the **intra-class distance** inside the class **decision boundary**. As a result, it may degrade downstream classification performance [1].
- In this study, we further hypothesize that **increasing intra-class distance may push samples closer to the decision boundary If this is true, those samples may become more vulnerable to small perturbations and thus more likely to be misclassified.**
- Therefore, this project experimentally investigates whether models pretrained with contrastive learning are more vulnerable to adversarial attacks such as FGSM.

## Experiment Setting
- We used **SimCLR** [2] as the contrastive learning method.
- As a comparison baseline, we used **RotNet** [3], an SSL method based on a pretext task.
- The backbone model was VGG11, and the optimizer was Adam.
- We used contrastive loss for contrastive learning and cross-entropy loss for linear evaluation.
- The mini-batch size was 512, and the learning rate was 0.001.
- Model performance was evaluated using the **linear evaluation** protocol [2]. After freezing the representation layer, we attached a linear classifier, trained only that layer, and measured the classification accuracy.
- The contrastive learning model was trained for 500 epochs, and the linear classifier was then trained in a supervised manner for 100 epochs.
- For adversarial evaluation, we used the **Fast Gradient Sign Method (FGSM)**.

## Experiment Results

### Linear Evaluation under FGSM Attacks

| Accuracy | ASR($\epsilon=0.03$)  | ASR($\epsilon=0.07$)  | ASR($\epsilon=0.1$)  |
|---|---|---|---|
| <img width="400" height="300" alt="finetune_accuracy" src="https://github.com/user-attachments/assets/b8e85573-b6c2-4776-89c3-60ee33f1d9a5" /> | <img width="400" height="300" alt="finetune_0 1" src="https://github.com/user-attachments/assets/75c4345e-a317-47e5-82ce-40c1f7875306" /> | <img width="400" height="300" alt="finetune_0 03" src="https://github.com/user-attachments/assets/9e18040a-e672-4963-8444-8893ed6f3bcd" /> | <img width="400" height="300" alt="finetune_0 07" src="https://github.com/user-attachments/assets/2d7d54f5-25d5-4fce-a8fb-936e3db1be86" /> |
|---|---|---|---|
| (a) | (b) | (c) | (d) |
| Accuracy with HN (e) | ASR($\epsilon=0.03$) with HN (f) | ASR($\epsilon=0.07$) with HN (g) | ASR($\epsilon=0.1$) with HN (h) |
|---|---|---|---|
| <img width="400" height="300" alt="finetune_HN_accuracy" src="https://github.com/user-attachments/assets/a14e1ed0-41cd-4f8a-9ac2-17bc55c3cf7e" /> | <img width="400" height="300" alt="finetune_HN_0 1" src="https://github.com/user-attachments/assets/2f0fbc5f-139a-49d6-9bcd-29119970bfe5" /> | <img width="400" height="300" alt="finetune_HN_0 03" src="https://github.com/user-attachments/assets/f00a7948-704e-40ad-88ed-e7fce1696729" /> | <img width="400" height="300" alt="finetune_HN_0 07" src="https://github.com/user-attachments/assets/38120149-bfd4-437d-bfd3-27f2fb9f0352" /> |
|---|---|---|---|
- **Attack Success Rate (ASR)**: the rate at which an adversarial attack successfully causes misclassification
- **$\epsilon$**: the perturbation magnitude in FGSM, i.e., the attack strength
- **Aug**: linear evaluation after learning the representation layer with augmented data
- **NoAug**: linear evaluation after learning the representation layer without augmented data
- **$\tau = 0.1, 0.3, 0.5$**: the temperature parameter used in contrastive learning
- **Hard Negative (HN)**: the original SimCLR uses augmented views rather than the original image itself. In the HN experiments, however, the non-augmented original image was also used.

### Intra-class Distance

| Setting | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck | mean | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aug | 0.28 | 0.30 | 0.34 | 0.30 | 0.31 | 0.31 | 0.26 | 0.34 | 0.29 | 0.30 | 0.303 | 0.26 | 0.34 |
| No_Aug | 0.19 | 0.23 | 0.21 | 0.20 | 0.20 | 0.23 | 0.20 | 0.23 | 0.20 | 0.21 | 0.210 | 0.19 | 0.23 |
| $\tau = 0.1$ | 4.87 | 4.63 | 5.05 | 5.09 | 4.79 | 5.15 | 4.72 | 5.06 | 4.80 | 4.50 | 4.866 | 4.50 | 5.15 |
| $\tau = 0.3$ | 3.84 | 3.59 | 4.08 | 4.05 | 3.75 | 3.97 | 3.74 | 3.85 | 3.75 | 3.39 | 3.801 | 3.39 | 4.08 |
| $\tau = 0.5$ | 3.09 | 2.86 | 3.39 | 3.29 | 3.06 | 3.24 | 3.10 | 3.21 | 3.02 | 2.69 | 3.095 | 2.69 | 3.39 |
| $\tau = 0.1$ with HN | 5.14 | 5.20 | 5.19 | 5.21 | 5.02 | 5.30 | 4.87 | 5.40 | 5.10 | 4.99 | 5.142 | 4.87 | 5.40 |
| $\tau = 0.3$ with HN | 4.00 | 3.83 | 4.10 | 4.06 | 3.80 | 4.07 | 3.81 | 4.14 | 3.79 | 3.59 | 3.919 | 3.59 | 4.14 |
| $\tau = 0.5$ with HN | 3.22 | 3.09 | 3.45 | 3.31 | 3.15 | 3.35 | 3.10 | 3.48 | 3.07 | 2.85 | 3.207 | 2.85 | 3.48 |

### Inter-class Distance

| Setting | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck | mean | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $\tau = 0.1$ | 2.4738 | 2.7410 | 2.1766 | 2.1403 | 2.3384 | 2.3518 | 2.6132 | 2.5669 | 2.5520 | 2.7479 | 2.47019 | 2.1403 | 2.7479 |
| $\tau = 0.3$ | 1.8628 | 2.0038 | 1.7325 | 1.6796 | 1.8669 | 1.7776 | 1.9749 | 1.9058 | 1.9348 | 1.9823 | 1.87210 | 1.6796 | 2.0038 |
| $\tau = 0.5$ | 1.6843 | 1.7701 | 1.6205 | 1.5293 | 1.6666 | 1.6076 | 1.8799 | 1.7392 | 1.7573 | 1.7838 | 1.70386 | 1.5293 | 1.8799 |
| $\tau = 0.1$ with HN | 2.6944 | 3.0501 | 2.3851 | 2.3631 | 2.5105 | 2.6067 | 2.8730 | 2.8137 | 2.8453 | 3.0502 | 2.71921 | 2.3631 | 3.0502 |
| $\tau = 0.3$ with HN | 1.9589 | 2.1585 | 1.7967 | 1.7550 | 1.8796 | 1.8505 | 2.0259 | 2.0681 | 2.0304 | 2.1095 | 1.96331 | 1.7550 | 2.1585 |
| $\tau = 0.5$ with HN | 1.8080 | 2.0432 | 1.7103 | 1.6560 | 1.7826 | 1.7554 | 1.9761 | 1.9264 | 1.9012 | 1.9529 | 1.85121 | 1.6560 | 2.0432 |

- When using SSL, the classification accuracy was lower than that of contrastive learning up to 50 epochs.
- However, after 50 epochs, the accuracy of contrastive learning improved only marginally, whereas SSL showed a relatively more stable improvement.
- At a low attack strength, such as $\epsilon = 0.03$, SSL showed a higher ASR.
- In contrast, when the attack strength increased to $\epsilon = 0.07$ or higher, contrastive learning showed a higher ASR, indicating greater vulnerability to adversarial attacks.
<br>

**The main finding of this study is as follows.**  
- We initially expected that using **HN**, that is, incorporating the original image, would increase the intra-class distance and therefore also increase the ASR. However, comparing Figures **(b)–(d)** and **(f)–(h)** shows that using HN actually reduced the ASR.
- As shown in the **Intra-class Distance** table above, using HN does indeed increase the **intra-class distance**. In other words, samples within the same class become farther apart.
- However, the **Inter-class Distance** table also shows that the **inter-class distance** increases as well.
- This can be explained by the training mechanism of contrastive learning. When HN is used, different images are treated as negative pairs even if they belong to the same class, which pushes them farther apart.
- However, if we assume a dataset with 10 classes, the probability that a sample is pushed away from a sample in the **same class** is **1/10**, whereas the probability that it is pushed away from a sample in a **different class** is **9/10**. Under the IID assumption, using original images can therefore enlarge not only the intra-class distance but also the inter-class distance even more substantially.
- As a result, contrary to our initial expectation, incorporating the original image increased robustness to adversarial attacks.

### t-SNE

| Aug (a) | $\tau = 0.1$ (b) | $\tau = 0.3$ (c) | $\tau = 0.5$ (d) |
|---|---|---|---|
| <img width="1052" height="749" alt="acc_aug" src="https://github.com/user-attachments/assets/edb4cc79-ac5f-4f28-a7b4-6973aaa19bb0" /> | <img width="1056" height="749" alt="01" src="https://github.com/user-attachments/assets/e144baa6-40e4-442d-9997-d1811f8529e6" /> | <img width="1057" height="751" alt="03" src="https://github.com/user-attachments/assets/3b4ffdcb-ef7e-495e-8277-7c0462f1661f" /> | <img width="1055" height="748" alt="05" src="https://github.com/user-attachments/assets/5b03d3ce-00e0-4617-b15c-b58bb62edc53" /> |

| NoAug (a) | $\tau = 0.1$ (b) with HN | $\tau = 0.3$ (c) with HN | $\tau = 0.5$ (d) with HN |
|---|---|---|---|
| <img width="1053" height="747" alt="acc_augX" src="https://github.com/user-attachments/assets/a31b15b3-5c99-4d76-a4b3-b8f76a814d58" /> | <img width="1058" height="746" alt="HN_01" src="https://github.com/user-attachments/assets/66d1bdc0-6d83-445f-983a-fe3714605840" /> | <img width="1051" height="748" alt="HN_03" src="https://github.com/user-attachments/assets/a5a9fd9f-9510-493f-b7f0-36215a3be509" /> | <img width="1053" height="748" alt="HN_05" src="https://github.com/user-attachments/assets/710ebac4-40d1-4ca0-89bb-73b33a7d6a94" /> |

## Future Work
- In future work, we plan to investigate whether contrastive learning methods that use **clustering** to alleviate the **false negative** problem are also vulnerable to adversarial attacks.
- It is also necessary to compare our results with a broader range of SSL methods beyond RotNet in order to evaluate the generality of the findings.
- In addition, future studies should propose concrete strategies to mitigate the limitations of contrastive learning in terms of adversarial robustness.

## References
[1] Huynh, Tri, et al. "Boosting contrastive self-supervised learning with false negative cancellation." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2022.  
[2] Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International Conference on Machine Learning. PMLR, 2020.  
[3] Gidaris, Spyros, Praveer Singh, and Nikos Komodakis. "Unsupervised representation learning by predicting image rotations." arXiv preprint arXiv:1803.07728 (2018).
