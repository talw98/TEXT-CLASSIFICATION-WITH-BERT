# TEXT CLASSIFICATION WITH BERT
Text Classification using BERT Framework



## Background

This project focuses on text classification using BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art pre-trained language model. The primary goal is to predict the publication year of COVID-19-related articles based on their abstracts, examining temporal changes in language patterns related to the pandemic.

## Data Processing

### Dataset

The dataset (covid_df.csv) consists of abstracts and publication years of COVID-19-related articles. To prepare the data for analysis, we perform the following steps:

1. **Data Splitting:**
   - Use the data_splitting_script.py to split the dataset into training, validation, and test sets.

2. **TF-IDF Classifier:**
   - Explore the impact of different vocabulary sizes using the TF-IDF classifier script (tfidf_classifier_script.py).
   - Determine the optimal max_features value for the TF-IDF vectorizer.

## Model Evaluation

### PubMedBert Text Classification

1. **BERT Text Classification:**
   - Utilize the BERT model to perform text classification on the dataset.
   - Run the bert_text_classification.py script to evaluate the performance of PubMedBert.

## Results

### Confusion Matrices

#### TF-IDF Classifier
```
[[5 6 1 0]
 [1 5 2 0]
 [0 6 6 0]
 [0 0 2 0]]
```

#### PubMedBert Text Classification
```
[[0 12 0 0]
 [0 8 0 0]
 [0 12 0 0]
 [0 2 0 0]]
```

## Conclusion

This project provides insights into text classification using BERT and TF-IDF on COVID-19-related articles. By exploring different models and preprocessing techniques, we gain a better understanding of the temporal language patterns surrounding the pandemic. 

## Model Accuracy

### TF-IDF Classifier
The confusion matrix suggests that the TF-IDF classifier achieved mixed results. Some classes (years) were predicted accurately, while others had misclassifications.

### PubMedBert Text Classification
The PubMedBert model, based on the confusion matrix, appears to have challenges accurately predicting the publication years. The model may need further tuning or exploration of different approaches to improve its performance.


