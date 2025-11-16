# Fake Job Post Detection using ML and NLP

Detecting fraudulent job postings is a critical task in online recruitment platforms. This project leverages **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques to classify job posts as either *real* or *fake*, helping protect job seekers from scams.

---

## 📌 Project Overview
- **Goal:** Build a predictive model that identifies fake job postings.
- **Approach:** Use NLP to process job descriptions and ML algorithms to classify them.
- **Dataset:** Publicly available job postings dataset (e.g., Kaggle Fake Job Posting dataset).
- **Outcome:** A web application that allows users to input job descriptions and receive predictions.

---

## ⚙️ Features
- Text preprocessing (tokenization, stopword removal, stemming/lemmatization).
- Feature extraction using TF-IDF and word embeddings.
- Classification models (Logistic Regression, Random Forest, Naive Bayes, etc.).
- Evaluation metrics: Accuracy, Precision, Recall, F1-score.
- Simple web interface for predictions.

---

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries:** Scikit-learn, Pandas, NumPy, NLTK, SpaCy, Matplotlib, Seaborn
- **Frameworks:** Flask / Streamlit (for web app)
- **Version Control:** Git & GitHub

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Venkatateja11/Fake-Job-Post-Detection-by-using-ML-and-NLP.git
cd Fake-Job-Post-Detection-by-using-ML-and-NLP
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py
```
or if using Streamlit:
```bash
streamlit run app.py
```

---

## 📊 Results
- Achieved high accuracy in detecting fake job posts.
- Models show strong performance in distinguishing fraudulent vs. genuine postings.
- Visualizations included for dataset insights and model evaluation.

---

## 📂 Repository Structure
```
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks for experiments
├── app.py              # Web application script
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
```

---

## 🤝 Contributing
Contributions are welcome! Please fork the repo and submit a pull request with improvements.

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author
Developed by **Venkatateja11**  
GitHub: [Venkatateja11](https://github.com/Venkatateja11)
