# Project-1: Text Retrieval

## Our Kaggle Implementation
https://www.kaggle.com/code/antonion01/final

## Introduction

Project 1 focuses on the task of text retrieval, which is the process of searching and returning relevant documents for a given query from a collection of documents. This project includes two main subtasks:

### Task 1: Document Full Ranking

We are able to rank documents based on their relevance to a given question. thanks to a document collection provided. This task simulates the construction of an end-to-end retrieval system.

### Task 2: Top-k Re-ranking

For Task 2, the goal is to re-rank these documents based on their relevance to the question. This scenario mirrors real-world situations where end-to-end systems are implemented as retrieval followed by top-k re-ranking. 

Full details of the project can be found in the [Report](./Report.pdf).

## Dataset

The corpus used contains a total of 1,471,405 documents, split into training and test sets as follows:

- Task 1 Train: 532,751 examples
- Task 2 Train: 10 examples
- Task 1 Test: 6,980 examples
- Task 2 Test: 33 examples

## Project Evaluation

The project was evaluated based on the following criteria:

### Results - Metrics

The evaluation score, S, is computed as a weighted average of the scores from these two tasks:
S = w1 * Recall@10 + w2 * NDCG   
Where w1 = 0.8 and w2 = 0.2.

## Constraints of project

- We can use either supervised or unsupervised methods for the project. However, for supervised methods, we must train the model using the provided labeled data, and are not allowed to use publicly available pre-trained models.
  
- It is prohibited to use the TF-IDF implementation from sklearn or any other libraries.
