# CLAVAS: Contrastive Learning – Adversarial Vulnerability And Susceptibility

## Acknowledgement
- 이프로젝트는 gachon 대학원의 Artificial Intelligence-based Image Recognition 수업에서 시작된 프로젝트이다.

## Abstract
- 최근 unlabeled data를 활용하여 모델의 representation layer를 학습시키는 기법으로 self-supervised learning(SSL) 또는 contrasitve larning (CL)을 많이 활용하고 있다. 
- 그러나 contrastive learning의 학습 방식은 실제로 동일한 클래스에 포함되지만, 다른 이미지라면 negative fair로 분류를 하여 학습을 시킨다.
- 그러나, 이러한 학습 방식이 class의 decision boundary내에서 이미지간의 intra distance를 멀게 만들어 실제로 down streamtask시 분류 성능을 감소시킨다[]. 
- 나는 intra class 거리를 늘린다는것은 샘플이 decision boundary 근처에 가깝도록 만들거라고 생각했다. 따라서, 이는 decision boundary에 가까운 샘플에 노이즈를 추가하여 miss classficiaton하도록 만드는 FGSM과 같은 adversial attack에 더 취약하게 만들것이라 의심했다.
- **이 프로젝트는 과연 contrastive learning 을 사용하여 모델을 pre traing 시켰을때, adverstial attack에 더 취약한지 가설을 검증한다.**

## Experiment Setting
- constrasitve learning: SimClar2[2]
- CL과의 비교를 위해 pretriang (pretext task)로 SSL인  RotNet[3]을 사용함
- 모델: vgg11, optimizer: Adam, loss: constrasive loss, cross entropy.
- mini batch: 512, learning rate": 0.001
- 모델 평가 팡식: linear evaluation[2]. represention layer를 얼린후, linear layer를 붙인뒤 linear layer를 학습시키고 정확도를 평가한다.
- contrastive learning 500 epoch 학습 시키고 linear layer는 supervised learning 방식으로 100 epoch 학습시킴
- adversial attack: Fast Gradient Sign Method (FGSM)

## experiment Result

### Linear evaluation under FGSM attacks

| Accuracy (a) | ASR($epsilon=0.03$) (b) | ASR($epsilon=0.07$) (c) | ASR($epsilon=0.1$) (d) |
| 이미지 1 | 이미지 2 | 이미지 3 | 이미지4 |
| Accuracy wit HN (e) | ASR($epsilon=0.03$) with HN (F) | ASR($epsilon=0.07$) with HN (g) | ASR($epsilon=0.1$)with HN (h)|
| 이미지 1 | 이미지 2 | 이미지 3 | 이미지4 |

- Attack Success Rate (ASR)
- epsilo: FGSM에서 공격의 정도
- Aug: 증간된 데이터와 SSL을로 represention layer를 학습 시키고 linear eval 함
- NoAug: 증강되지 않는 데이터로 SSL로 represnetion layer를 학습시키고 linear eval함
- tau 0.1, 0.3, 0.5: ccontrastive leanring 적용. contrasitve learning 의 tau값.
- Hard Negative (HN): 원래 simclr은 원본 이미지를 사용하지 않고 증강된 이미지를 사용한다. 하지만 HN을 사용한 실험은 증강되지 않은 원본 이미지를 사용한것이다.

### intra class distacne

-> intra 거리 표 만들것
표1

### inter class distance

-> inter 거리 표 만들것
표2

- SSL을 사용한경우 50epoch 까지는 정확도가 CL을 사용했을때 보다 낮음. 하지만 50epoch 이상부터는 CL이 정확도가 많이 오르지 못함.
- $epsilon=0.03$ 즉 공격류이 낮을때는 오히려 SSL이 공격 ASR이 높음 하지만 $epsilon=0.07$ 와 같이 공격 노이즈가 증가하면 CL이 ASR이 더 높음
- **Key insight! 원래는 HN 즉, 원본 이미지를 사용하면 intra class distance를 멀게 하여 ASR이 증가할줄 알았음. 하지만 그림 b-d 그리고 f-h를 비교해보면 HN을 사용했을때 ASR이 오히려 감소함. 표1을 보면 확실이 intra class 거리가 증가함. 즉 클래스의 decision boundary안에서 샘플간 거리가 증가함. 그러나 표2를 보면, 클래스간 거리 역시 증가함. 그 이유는 HN을 사용하면 실제로 같은 클래스라도 CL에서는 negative pair로 취급해 다른이미지라면 거리가 멀어짐. 그러나, 예를 들어 10개의 클래스를 학습한다고 했을때 확률적으로 같은 클래스 끼리 멀어질 확률은 1/10이고 다른 클래스와 멀어질 확률은 9/10임, 데이터셋이 iid하다는 가정하에.-**
- ** 결론 적으로 원본 이미지를 사용하면 오히려 adversial attack에 대한 강건성이 늘어남**

### t-SNE

| Accuracy (a) | ASR($epsilon=0.03$) (b) | ASR($epsilon=0.07$) (c) | ASR($epsilon=0.1$) (d) |
| 이미지 1 | 이미지 2 | 이미지 3 | 이미지4 |
| Accuracy wit HN (e) | ASR($epsilon=0.03$) with HN (F) | ASR($epsilon=0.07$) with HN (g) | ASR($epsilon=0.1$)with HN (h)|
| 이미지 1 | 이미지 2 | 이미지 3 | 이미지4 |

## Furtur Work

- simclr왜에도 HN문제를 해결하기위해 클러스터링을 기반으로 하는 CL 기법들 역시 adversial attack에 취약한 문제가 있는지 탐구하고자 함.
- 또한 SSL 역시 다른 기법을 사용해 비교 실험이 필요해 보임
- 또한 명확한 CL이 adverserial attack에 대응하기 위한 해결책을 제시할 필요가 잇음.

## References
[1] Huynh, Tri, et al. "Boosting contrastive self-supervised learning with false negative cancellation." Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2022.
[2] Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International conference on machine learning. PmLR, 2020.
[3] Gidaris, Spyros, Praveer Singh, and Nikos Komodakis. "Unsupervised representation learning by predicting image rotations." arXiv preprint arXiv:1803.07728 (2018).
