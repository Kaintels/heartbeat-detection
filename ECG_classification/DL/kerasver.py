import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Conv1D, GRU, LSTM, Activation
from tensorflow.keras.layers import MaxPooling1D, Softmax, AveragePooling1D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.utils import to_categorical

y_tests = to_categorical(y_test, 2)
y_trains = to_categorical(y_train, 2)

batch_sizes=16

model = Sequential()

model.add(Conv1D(filters=32, padding='same', kernel_size=3, input_shape=(510,1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=32, padding='same', kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=64, padding='same', kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=64, padding='same', kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=128, padding='same', kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=128, padding='same', kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=256, padding='same', kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=256, padding='same', kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(LSTM(units=64, return_sequences=True))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy',  metrics=["accuracy"])
history = model.fit(x_train, y_trains, shuffle=True, batch_size=batch_sizes, epochs=1,verbose=1)
history_dict = history.history
loss, accuracy = model.evaluate(x_test, y_tests, verbose=1, batch_size=batch_sizes)
pred = model.predict(x_test, batch_size=batch_sizes, verbose=1)
predicted_classes = np.argmax(pred, axis=1)
target_names = ['Non-ectopic', 'Ectopic']


from sklearn.metrics import classification_report

print(classification_report(y_test, predicted_classes,target_names=target_names, digits = 6))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predicted_classes))