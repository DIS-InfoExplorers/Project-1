![Project Logo](Logo.png)
# Project-1: Text Retrieval

## Introduction

Project 1 focuses on the task of text retrieval, which is the process of searching and returning relevant documents for a given query from a collection of documents. This project includes two main subtasks:

### Task 1: Document Full Ranking

We are able to rank documents based on their relevance to a given question. You thanks to a document collection provided. This task simulates the construction of an end-to-end retrieval system.

### Task 2: Top-k Re-ranking

For Task 2, you will be provided with an initial ranking of documents per question. Your goal is to re-rank these documents based on their relevance to the question. This scenario mirrors real-world situations where end-to-end systems are implemented as retrieval followed by top-k re-ranking. Task 2 allows you to focus on the re-ranking aspect without the need to implement a complete end-to-end system.

## Dataset

The corpus used contains a total of 1,471,405 documents, split into training and test sets as follows:

- Task 1 Train: 532,751 examples
- Task 2 Train: 10 examples
- Task 1 Test: 6,980 examples
- Task 2 Test: 33 examples

## Project Evaluation

Your project will be evaluated based on the following criteria:

### Results - Metrics

The evaluation score, S, is computed as a weighted average of the scores from these two tasks:
S = w1 * Recall@10 + w2 * NDCG

Where w1 = 0.8 and w2 = 0.2. This criterion accounts for 75% of your grade in the Results section. The grading further depends on your performance relative to other teams, with the top 10% receiving the highest score.

## Constraints of project

- You can use either supervised or unsupervised methods for the project. However, for supervised methods, you must train the model using the provided labeled data, and you are not allowed to use publicly available pre-trained models.
  
- You are prohibited from using the TF-IDF implementation from sklearn or any other libraries.
