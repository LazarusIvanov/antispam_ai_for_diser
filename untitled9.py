# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hLlXjAvT1NExmpF1MSk4THNuqs7Icc7l
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np #6

# Определение функции для классификации сообщения
def classify_message(message, model):
  # Предсказание вероятностей принадлежности сообщения к классам
    prediction = model.predict_proba([message])
    spam_probability = prediction[0][1]
    result = 'спам' if prediction[0][1] > prediction[0][0] else 'не спам'
    return result, spam_probability #7

# чтение данных
data = pd.read_csv('spam.csv', delimiter=',', header=None, encoding='ISO-8859-1', skiprows=1)
X = data[1] #текст сообщения
y = data[0] #метка сообщения #4

# Преобразование меток в числовые значения с помощью LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y) #3

# Создание и обучение модели
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
model.fit(X, y) #6

# Ввод нового сообщения
new_message = input("Введите ваше сообщение - ") #2

# Классификация сообщения и расчет вероятности спама
result, spam_probability = classify_message(new_message, model) #2

# Вывод результатов
print(f"Письмо '{new_message}' классифицировано как: {result}")
print(f"Вероятность спама: {spam_probability:.2f}") #3 (29)