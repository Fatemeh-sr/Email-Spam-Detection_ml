# Spam Email Detection using Machine Learning

This project implements a **spam email classifier** using two machine learning models:

* **Multilayer Perceptron (MLP) Classifier**
* **Multinomial Naive Bayes**

The system is trained on the **Enron Spam Dataset** and uses **TF-IDF vectorization** to convert email text into numerical features. It can also classify new email text files as **spam** or **ham**.

---

## Features

* Text preprocessing using **TF-IDF**
* Binary classification: **Spam vs Ham**
* Model comparison (MLP vs Naive Bayes)
* Performance evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* Predicts spam/ham for new email text files

---

## Project Structure

```
.
├── enron_spam_data.csv
├── new_email.txt
├── new_email2.txt
├── spam_classifier.py
└── README.md
```

---

## Models Used

### 1. Multilayer Perceptron (MLP)

* Early stopping enabled
* Hidden layer neural network
* Suitable for learning complex patterns in text data

### 2. Multinomial Naive Bayes

* Fast and efficient for text classification
* Well-suited for TF-IDF features

---

 The program will:

   * Train both the MLP and Naive Bayes models
   * Print evaluation metrics (Accuracy, Precision, Recall, F1-score)
   * Predict whether any given input email (e.g., new_email.txt, new_email2.txt) is spam or ham

---

## Dataset

* **Enron Spam Dataset**
* Columns used:

  * `Subject`
  * `Message`
  * `Spam/Ham` (target label)
