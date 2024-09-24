# SecFormer: Fast and Accurate Privacy-Preserving Inference for Transformer Models via SMPC.
[**Paper**](https://aclanthology.org/2024.findings-acl.790.pdf) | 
[**Usage**](#usage) |
[**Citation**](#citation) |

This repository includes code for our paper "SecFormer: Fast and Accurate Privacy-Preserving Inference for Transformer Models via SMPC". It consists of three folders.

## Usage
- Performance: For reproduce the performance evaluation results of privacy-preserving inference in Table 2 of this paper.
- Efficiency: For reproduce the efficiency evaluation results of privacy-preserving inference in Table 3 of this paper.
- Operation: For reproduce evaluation results of Figure 5-9. Which is the time and communication overhead statistics of the SMPC protocols for GeLU, LayerNorm and Softmax.
The execution steps of the experiments are described in the REAMD.md file in each folder.[text](vscode-local:/d%3A/Edge_download/MPCFormer_README.md)

## Citation
If you find this repository useful, please cite our paper using

````
@inproceedings{luo2024secformer,
  title={SecFormer: Fast and Accurate Privacy-Preserving Inference for Transformer Models via SMPC},
  author={Luo, Jinglong and Zhang, Yehong and Zhang, Zhuo and Zhang, Jiaqi and Mu, Xin and Wang, Hui and Yu, Yue and Xu, Zenglin},
  booktitle={Findings of the Association for Computational Linguistics ACL 2024},
  pages={13333--13348},
  year={2024}
}
````
