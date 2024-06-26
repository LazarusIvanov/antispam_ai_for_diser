import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model

data = pd.read_csv('…', delimiter=',', header = None, encoding='ISO-8859-1', skiprows= 1)
tokenizer = Tokenizer(num_words=10000, oov_token="")
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)
padded_sequences = pad_sequences(sequences, padding='post')
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y, num_classes=2)
model = Sequential([
Embedding(100000, 31, input_length=padded_sequences.shape[1]),
LSTM(31),
Dense(31, activation='relu'),
Dense(64, activation='relu'),
Dense(2, activation='softmax')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_sequences, y, epochs=5, validation_split=0.15)
def classify_message(message):
sequence = tokenizer.texts_to_sequences([message])
padded = pad_sequences(sequence, padding='post')
prediction = model.predict(padded)
value1 = prediction
result = "спам" if prediction[0][0] > prediction[0][1] else "не спам"
return result
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
while True:
new_message = input("письмо на проверку - ")
print(f"{new_message} - {classify_message(new_message)}")

def classify_message(message, model):
prediction = model.predict_proba([message])
spam_probability = prediction[0][1]
result = 'спам' if prediction[0][1] > prediction[0][0] else 'не спам'
return result, spam_probability
data = pd.read_csv('…', delimiter=',', header=None, encoding='ISO-8859-1', skiprows=1)
X = data[1] #текст сообщения
y = data[0]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
model = Pipeline([
('vectorizer', CountVectorizer()),
('classifier', MultinomialNB())
])
model.fit(X, y)
new_message = ("ваше сообщение - ")
result, spam_probability = classify_message(new_message, model)
value2 = spam_probability
print(f"Письмо '{new_message}' классифицировано как: {result}")
print(f"Вероятность спама: {spam_probability:.2f}")

if value1 < 0.5 and value2 < 0.5:
print("Не спам")
elif value2 > 0.5 and value2 > 0.5:
print("Спам")
else:
print("Спам")
