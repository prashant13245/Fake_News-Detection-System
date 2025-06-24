# 📰 Fake News Detection using Machine Learning

This project uses multiple machine learning algorithms to classify news articles as **Fake** or **Not Fake** based on their content. The dataset contains labeled news headlines, and the model is trained to identify deceptive news patterns.

---

## 📁 Dataset

Two datasets are used:

- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains genuine (real) news articles

Each dataset includes columns like:
- `title`
- `text` (main article content)
- `subject`
- `date`

---

## 🛠️ Preprocessing Steps

- Loaded and labeled both datasets (`class = 0` for fake, `class = 1` for true)
- Dropped unnecessary columns: `title`, `subject`, `date`
- Removed HTML tags, punctuation, digits, and stopwords
- Applied custom text cleaning function (`wordopt`)
- Vectorized text using **TF-IDF (TfidfVectorizer)**

---

## 🤖 Machine Learning Models Used

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Gradient Boosting Classifier**
4. **Random Forest Classifier**

Each model is trained using a **75-25 train-test split** and evaluated using:

- Accuracy score
- Classification report (precision, recall, F1-score)

---

## 🧪 Manual Testing

A custom function `manual_testing(news)` allows you to input any news text and get predictions from all four models.

```python
manual_testing("Your news text goes here...")



✅ Requirements
Python 3.x

pandas

numpy

scikit-learn

seaborn

matplotlib

| Model               | Accuracy (Approx.) |
| ------------------- | ------------------ |
| Logistic Regression | 98%                |
| Decision Tree       | 99%                |
| Gradient Boosting   | 99%                |
| Random Forest       | 99%                |



📌 How to Use
Download or clone the repository.

Place Fake.csv and True.csv in the same directory.
Run the Jupyter notebook:
jupyter notebook
Open Fake News Detection.ipynb and execute all cells.
