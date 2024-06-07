Vulnerability Detection with Feature Fusion and Learnable Edge-type Embedding Graph Neural Network

<p aligh="center"> This repository contains the code and data for <b>Vulnerability Detection with Feature Fusion and Learnable Edge-type Embedding Graph Neural Network</b> </p>

## Introduction

Deep learning methods are widely employed in vulnerability detection, and graph neural networks have shown effectiveness in learning source code representation. However, current methods overlook non-relevant noise information in the code property graph and lack specific graph neural networks designed for code property graph.To address these issues, this paper introduces Leev, an automated vulnerability detection method. We developed a graph neural network tailored to the code property graph, assigning iterative vectors to diverse edge types and integrating them into the message passing between nodes to enable the model to extract hidden vulnerability information.In addition, virtual nodes are incorporated into the graph for feature fusion, mitigating the impact of irrelevant features on vulnerability information within the code.Specifically, for the FFMPeg+Qemu, Reveal, and Fan et al. datasets, the F1 metrics exhibited improvements of 7.02\%, 21.69\%, and 27.74\% over the best baseline, correspondingly.lts.

----------

## Contents
1. [Dataset](#Dataset)
2. [Code Property Graph](#Code-Property-Graph)
3. [Preprocessing](#Preprocessing)
4. [Requirement](#Requirement)
5. [Code](#Code)
6. [Reference](#Reference)

## Dataset

The Dataset we used in the paper:

Fan et al.[1]: https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing

Reveal [2]: https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy

FFMPeg+Qemu [3]: https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF

## Code Property Graph

In this study, we use Joern to generate Code Property Graph,The version we used is v1.0.141, which can download in https://joern.io/

## Preprocessing
In order to unify the data format, before running the Fan et al and FFMPeg+Qemu datasets, we should cd pre_code and python make_code_FFmp.py or python make_code_FAN.py

## Requirement

Please check all requirements in the requirement.txt

## Code

1. python create_graphs.py. Don't forget to modify the JOERNPATH in line 2 and dataset in line 3
 
2. Modify the dataset in line 294 and python data_preprocess.py.

3. python main.py LeeV VDM data/fin_code_Reveal

Because of the randomness in the deep learning model and the different data splitting, the model performance may be different from the results reported in the paper.

## Reference

[1] Jiahao Fan, Yi Li, Shaohua Wang, and Tien Nguyen. 2020. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. In The 2020 International Conference on Mining Software Repositories (MSR). IEEE.

[2] Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray. 2020. Deep Learning based Vulnerability Detection: Are We There Yet? arXiv preprint arXiv:2009.07235 (2020).

[3] Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems. 10197–10207.
