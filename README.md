# Fake_News_Detector

# 📰 Fake News Detection using NLP

This project uses Natural Language Processing (NLP) techniques to detect whether a news article is **Fake** or **Real**. It is built using Python and machine learning libraries such as **scikit-learn** and **nltk**.

---

## 📌 Project Overview

- **Goal**: Classify news articles as fake or real based on their content.
- **Approach**: Data preprocessing, TF-IDF vectorization, and training a Passive Aggressive Classifier.
- **Language**: Python
- **Libraries**: `pandas`, `nltk`, `sklearn`

---

## 📁 Dataset

We have used publicly available fake/real news datasets in CSV format.

### 🔗 Google Drive Dataset Links

- [🟢 Dataset Link 1](https://drive.google.com/file/d/1D5Pl8HgqCZoVJiHsVhcpA0vPFQIRWPEM/view?usp=drive_link)
- [🟢 Dataset Link 2](https://drive.google.com/file/d/1iAY8eIPci5GcjwvWsqJMHT4xr4rL8fmM/view?usp=drive_link)

Please download and place the `.csv` files in the project directory.

---

## 🧠 Model Details

| Step                | Description                                      |
|---------------------|--------------------------------------------------|
| Preprocessing        | Lowercasing, punctuation & stopword removal     |
| Vectorization        | TF-IDF (Term Frequency-Inverse Document Frequency) |
| Classifier           | Passive Aggressive Classifier (from `sklearn`)  |
| Output               | Prediction: `FAKE` or `REAL`                    |

---

## 🛠️ How to Run

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn nltk

