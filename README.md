# 🫀 CardioTree Classifier

A machine learning project that predicts the likelihood of heart disease using a Random Forest Classifier trained on the **Indicators of Heart Disease** dataset. This project demonstrates data preprocessing (label encoding and one-hot encoding), model training and evaluation, and visualization of decision trees.

---

## 📂 Project Structure

CardioTree-Classifier/

│

├── main.py                   # Main script for data preprocessing, model training, and visualization

├── heartDisease_2020_sampling.csv  # Dataset with heart disease indicators

├── utilities.py              # Helper functions for label encoding and one-hot encoding

├── README.md                 # Documentation and project overview

---

## 📊 Dataset

This project uses the *Indicators of Heart Disease Dataset* (2020), which contains health-related attributes such as:

- **Demographics:** Age category, sex, race  
- **Lifestyle factors:** Smoking, alcohol drinking, physical activity  
- **General health indicators:** BMI, sleep time, physical health, mental health  
- **Target Variable:** HeartDisease (Yes/No)

Dataset source: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

---

## 🛠 Features

✅ **Data Preprocessing**
- Label encoding of categorical variables (e.g., HeartDisease, Smoking, Sex)
- One-hot encoding for race categories

✅ **Model Building**
- Random Forest Classifier with balanced class weights
- Train-test split for model evaluation

✅ **Evaluation Metrics**
- Accuracy score on training and test data
- Confusion matrix for performance visualization

✅ **Decision Tree Visualization**
- Displays how individual decision trees in the random forest make classification decisions

✅ **Real-world Application Examples**
- Explores potential uses of decision trees in recommendation systems (e.g., book recommendations on Goodreads/Amazon)

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/CardioTree-Classifier.git
   cd CardioTree-Classifier
   ```
2. **Install dependencies**
```bash
  pip install pandas scikit-learn matplotlib
```
3. **Run the main program**
```bash
  python main.py
```
4. **Follow the interactive prompts in the console to explore the dataset and model.**
