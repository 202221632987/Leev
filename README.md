A Learnable Edge-type Embedding Model for Vulnerability Detection

<p aligh="center"> This repository contains the code and data for <b>A Learnable Edge-type Embedding Model for Vulnerability Detection</b> </p>

## Introduction

In recent years, deep learning-based methods to detect vulnerabilities have been widely used to improve Software security. These methods usually extract structural information, such as code property graphs from source code, and use neural networks, such as graph neural networks (GNNs), to learn the graph representation. However, these methods, while useful, do not account for the heterogeneous relationships that exist between edges and node types in code property graphs thus reducing graph representation learning performance. Moreover, existing models suffer from feature fusion deficiency, hindering them from fully aggregating node features to obtain the overall features of the graph. To address these issues, the paper proposes a vulnerability detection network named LeeV. LeeV uses a graph attention neural network with learnable edge embedding to capture edge heterogeneity and incorporates a virtual node connected to all nodes to achieve feature fusion. We evaluated LeeV on three publicly available C/C++ code vulnerability datasets, and the experimental results demonstrated that the F1 score improved by 1.63%, 23.05%, and 13.11%, compared to the best baseline results.

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
