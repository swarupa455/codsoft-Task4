# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset

df = pd.read_csv("/content/spam.csv", encoding='latin1')
df = df[['v1', 'v2']]  # Select relevant columns
df.columns = ['label', 'text']

# Preprocess the data
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

df['text'] = df['text'].apply(preprocess_text)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_predictions = nb_model.predict(X_test_tfidf)

# Train a Logistic Regression classifier
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_predictions = lr_model.predict(X_test_tfidf)

# Train a Support Vector Machine classifier
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
svm_predictions = svm_model.predict(X_test_tfidf)

# Evaluate the models
def evaluate_model(predictions, y_test):
    accuracy = accuracy_score(y_test, predictions)
    precision_spam = precision_score(y_test, predictions, pos_label='spam')
    recall_spam = recall_score(y_test, predictions, pos_label='spam')
    f1_spam = f1_score(y_test, predictions, pos_label='spam')

    precision_ham = precision_score(y_test, predictions, pos_label='ham')
    recall_ham = recall_score(y_test, predictions, pos_label='ham')
    f1_ham = f1_score(y_test, predictions, pos_label='ham')

    return {
        'accuracy': accuracy,
        'precision_spam': precision_spam,
        'recall_spam': recall_spam,
        'f1_spam': f1_spam,
        'precision_ham': precision_ham,
        'recall_ham': recall_ham,
        'f1_ham': f1_ham
    }

print("Naive Bayes Evaluation:")
print(evaluate_model(nb_predictions, y_test))

print("Logistic Regression Evaluation:")
print(evaluate_model(lr_predictions, y_test))

print("SVM Evaluation:")
print(evaluate_model(svm_predictions, y_test))

# Detailed classification report for further insights
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions, target_names=['ham', 'spam']))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions, target_names=['ham', 'spam']))

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions, target_names=['ham', 'spam']))
