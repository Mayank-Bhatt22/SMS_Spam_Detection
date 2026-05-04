# SMS_Spam_Detection

SMS Spam Detection app that classifies messages as Spam or Not Spam using NLP and machine learning. It preprocesses text, applies TF-IDF vectorization, and uses a trained model to predict results. Built with Python, scikit-learn, and Streamlit for real-time predictions.

# 📩 SMS Spam Detection 🚀

A machine learning-based web application that classifies SMS messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) techniques and a trained model.

---

## 🔍 Overview

Spam messages are a major issue in digital communication. This project builds an intelligent system that automatically detects spam messages using text preprocessing, feature extraction, and machine learning.

---

## ⚙️ Features

- 🔤 Text preprocessing (lowercase, tokenization, stopword removal, stemming)  
- 📊 TF-IDF vectorization for feature extraction  
- 🤖 Machine Learning model (Multinomial Naive Bayes)  
- 🌐 Interactive web app using Streamlit  
- ⚡ Real-time prediction of SMS/email messages  

---

## 🧠 Tech Stack

- Python  
- Pandas, NumPy  
- NLTK  
- Scikit-learn  
- Streamlit  

---

## 📂 Project Structure

SMS_Spam_Detection/
│── app.py
│── model.pkl
│── vectorizer.pkl
│── requirements.txt
│── spam.csv

---

## 🔄 Workflow

1. Data Cleaning & Preprocessing  
2. Text Transformation (Tokenization + Stemming)  
3. Feature Extraction using TF-IDF  
4. Model Training (Naive Bayes)  
5. Model Deployment using Streamlit  

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/Mayank-Bhatt22/SMS_Spam_Detection
cd SMS_Spam_Detection
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Example

Input:
Congratulations! You won ₹5000. Click here to claim.
Output:
🚨 Spam

---

## 📈 Future Improvements

Use advanced models (Logistic Regression, SVM, Deep Learning)
Improve accuracy with better preprocessing
Add multilingual support

---

## 👨‍💻 Author

## Mayank Bhatt
GitHub: https://github.com/Mayank-Bhatt22

Linkedin: https://linkedin.com/in/mayank-bhat

---

## ⭐ If you like this project
## Give it a ⭐ on GitHub!
