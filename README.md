# Stance Detection Using Multi-lingual Large Language Models

This repository contains codes to fine-tune four multi-lingual language models, namely MBERT, XLM-RoBERTa, DistilmBERT, and MDeBERTa on the task of stance detection.

## Models:

* **MBERT:** A multi-lingual BERT model.
* **XLM-RoBERTa:** A cross-lingual RoBERTa model.
* **DistilmBERT:** A distilled version of the multi-lingual BERT model.
* **MDeBERTa:** A multi-lingual DeBERTa model.

## Dataset:
The fine-tuning were conducted on Google Colab (T4 GPU) using a AraStance (Alhindi et al., [2021](https://aclanthology.org/2021.nlp4if-1.9/)) dataset.

## Evaluation Metrics:

* **Accuracy:** the ratio of the number of correct predictions to the total number of predictions.
* **F1-Score:** the harmonic mean of precision and recall.
* **Macro F1-score:** average of per-class f1-scores.

## Results and Analysis:

The following results are based on a single training run. This clearly indicates that the reported performance might vary slightly due to the inherent randomness in the training process.

### Validation Results

| Model | Accuracy | Agree | Disagree | Discuss | Unrelated | Macro f1-score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| MBERT | **0.826** | **0.794** | 0.722 | **0.605** | **0.906** | **0.757** |
| XLM-RoBERTa | 0.798 | 0.783 | **0.800** | 0.424 | 0.886 | 0.723 |
| DistilmBERT | 0.750 | 0.720 | 0.658 | 0.500 | 0.849 | 0.682 |
| MDeBERTa | 0.754 | 0.728 | 0.671 | 0.053 | 0.882 | 0.584 |

### Testing Results

| Model | Accuracy | Agree | Disagree | Discuss | Unrelated | Macro f1-score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| MBERT | **0.845** | **0.828** | 0.772 | **0.477** | 0.917 | **0.748** |
| XLM-RoBERTa | 0.842 | 0.801 | **0.777** | 0.468 | **0.935** | 0.745 |
| DistilmBERT | 0.799 | 0.817 | 0.623 | 0.389 | 0.903 | 0.683 |
| MDeBERTa | 0.785 | 0.751 | 0.667 | 0.075 | 0.906 | 0.600 |

## Future Work:

1. **Multiple Runs:** To obtain more robust results, consider running multiple training sessions with different random seeds and averaging the performance across the runs.
2. **Explore other multilingual models:** Experiment with other state-of-the-art multilingual models.
3. **Investigate data augmentation techniques:** Explore techniques to improve data diversity and model robustness.
4. **Fine-tune on larger and more diverse datasets:** Train the models on larger and more diverse datasets to enhance their generalizability.
 
In addition, it is possible to further improve the performance of the models on this classification task by carefully considering the following:

5. **Class Imbalance**: Techniques like class weighting or oversampling could be explored to address the class imbalance in the dataset.
6. **Hyperparameter Tuning**: Conduct a thorough hyperparameter search to optimize the performance of each model.

## Software/Libraries:

- Python 3.10.12
- NumPy 1.26.4
- PyTorch 2.5.1+cu121
- Transformers 4.46.3
- Scikit-learn 1.5.2
