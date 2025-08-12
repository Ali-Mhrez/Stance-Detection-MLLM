# Repository of the 5th chapter of my dissertation: [Stance Detection Toward Fake News Using Multi-lingual Transformers](https://github.com/Ali-Mhrez/Dissertation/blob/main/dissertation.pdf), pages 65--83

This repository contains the code for a Ph.D. research paper that systematically investigates the performance of several multilingual models on the task of stance detection in the context of fake news. The task involves classifying the relationship between a news article's body text and its headline into one of four categories: agree, disagree, discuss, or unrelated. The fine-tuned models include mBERT, xlm-RoBERTA, Distil-mBERT, and mDeBERTa.

## Goal and Background:

The increasing prevalence of code-switching and multilingual text in modern communication highlights a critical need for robust natural language processing (NLP) models capable of handling diverse linguistic inputs. Multilingual transformers have emerged as a powerful solution, offering the ability to analyze and understand text written in multiple languages and even transfer knowledge to low-resource languages like Arabic. Despite this potential, there is a significant research gap concerning the performance of these models on specific, high-stakes tasks like fake news stance detection in Arabic. The effectiveness and limitations of leading multilingual models on this task remain largely unexplored, which hinders progress and a deeper understanding of their capabilities.

This study aims to bridge this gap by conducting a thorough analytical investigation into the performance of state-of-the-art multilingual transformers. Our research is guided by the following key questions:
1. **Effectiveness:** How do modern multilingual transformers perform on the challenging task of Arabic fake news stance detection?
2. **Linguistic Challenges:** What are the specific linguistic challenges inherent in this task, and how do multilingual models cope with them?
3. **Dataset Limitations:** What are the limitations of the available datasets, and how can they be improved to better facilitate research in this area?

This research makes the following specific contributions to the field:
1. **Performance Evaluation:** We provide a systematic evaluation of several previously unstudied multilingual transformers on an Arabic stance detection task.
2. **Comparative Analysis:** We compare the performance of the models across various aspects, including their linguistic knowledge and robustness, to highlight their strengths and weaknesses.
3. **Error Analysis:** We analyze the classification errors of the studied models to identify the key challenges and failure modes.
4. **Future Directions:** We propose and shed light on potential approaches for overcoming the identified challenges, paving the way for more effective models in future work.

## Dataset
The [AraStance](https://aclanthology.org/2021.nlp4if-1.9/) and [UnifiedFC](https://aclanthology.org/N18-2004/) datasets include article bodies, headlines, and a corresponding class label. The label indicates the stance of the article body with respect to the headline. The article body can either Agree (AGR) or Disagree (DSG) with the headline, it can Discuss (DSC) it or be completely Unrelated (UNR).
| Data Source | Data Type | Instances | AGR | DSG | DSC | UNR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| [repo](https://github.com/Tariq60/arastance) | News articles | 4,063 | 25.1% | 11.0% | 9.5% | 54.3% |
| [repo](https://alt.qcri.org/resources/arabic-fact-checking-and-stance-detection-corpus/) | News articles | 3,042 | 15.6% | 2.9% | 13.4% | 68.1% |

## Data Preprocessing
AraStance is already divided into: Training, Validation, Testing sets.  
UnifiedFC is splitted into five folds, we use the first three for training, the fourth for validation, and the fifth for testing.  
No Special preprocessing was conducted except tokenization using the default tokenizer of each model.

## Models

| Model | Layers | Hidden Dimension | Attention Heads | Vocabulary Size | Training Tasks | Training Data | Languages | Parameters |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md) | 12 | 768 | 12 | 110K | MMLM+NSP | Wikipedia | 104 | 172M |
| [xlm-RoBERTa](https://aclanthology.org/2020.acl-main.747/) | 12 | 768 | 12 | 250K | MMLM | CC-100 | 100 | 270M |
| [Distil-mBERT](https://arxiv.org/abs/1910.01108) | 6 | 768 | 12 | 110K | KD+MMLM | Wikipedia | 104 | 134M |
| [mDeBERTa](https://arxiv.org/abs/2111.09543) | 12 | 768 | 12 | 250K | RTD | CC-100 | 100 | 276M |

## Hyperparameters

| Sequence Length | Batch Size | Warmup Ratio | Early Stopping | Loss | Optimizer |
|:---|:---:|:---:|:---:|:---:|:---:|
| 512 | 32 | 10% | 3 | Cross Entropy | Adam |

| Dataset/Model | mBERT | xlm-RoBERTa | Distil-mBERT | mDeBERTa
|:---|:---:|:---:|:---:|:---:|
| [AraStance](https://aclanthology.org/2021.nlp4if-1.9/) | 2e-5 | 3e-5 | 3e-5 | 3e-5 |
| [UnifiedFC](https://aclanthology.org/N18-2004/) | 4e-5 | 3e-5 | 5e-5 | 2e-5 |

## Key Results

The following are the key findings from our analysis of multilingual transformers on the Arabic stance detection task:
1. **Performance Convergence:** Despite their advanced architecture, the overall effectiveness of multilingual transformers showed a convergence with baseline deep learning models. This was especially pronounced under common challenges like limited dataset size and significant class imbalance.
2. **MBERT's Strengths:** MBERT demonstrated exceptional robustness and a strong ability for cross-lingual knowledge transfer, outperforming other multilingual models in our comparative analysis.
3. **Translation for Improvement:** Our experiments indicate that a key method for boosting the performance of multilingual transformers is through data augmentation via translation, which provides a valuable avenue for improving models in low-resource settings.

## Requirements

- Python 3.10.12
- NumPy 1.26.4
- PyTorch 2.5.1+cu121
- Transformers 4.46.3
- Scikit-learn 1.5.2

## Citation
```bash
@incollection{amhrez-mlm,
author = {Mhrez, ali; Ramadan, Wassim; Abo Saleh, Naser},
title = {Stance Detection Toward Fake News Using Multi-lingual Transformers},
booktitle = {Stance Detection in Natural Language Texts Using Deep Learning Techniques},
publisher = {University of Homs},
chapter = {5},
pages = {65--83},
year = {2024},
url = {https://github.com/Ali-Mhrez/Dissertation},
note = {This chapter is based on a dissertation submitted in partial fulfillment of the requirements for the degree of Doctor of Philosophy.}
}
```
