#!/usr/bin/env python
# coding: utf-8

#%%
import keras
keras.__version__

#
import pandas as pd
import numpy as np
#dataset = df.values

# ## Набор данных


data = pd.read_csv('merc.csv')


nnpDf = data.sort_values("price",ascending = False).iloc[131:]

nnpDf.drop(['transmission', 'fuelType', 'model'], axis=1, inplace=True)

df = nnpDf


y = df[['price']].values
x = df.drop('price', axis=1).values

from sklearn.model_selection import train_test_split
train_data, test_data, train_targets, test_targets = train_test_split(x, y, test_size = 0.2, random_state = 0)

print(train_data.shape,train_targets.shape,test_data.shape,test_targets.shape)


# вычисление среднего значения по всем обучающим данным
mean = train_data.mean(axis=0)
# вычитаем среднее значение
train_data -= mean
# вычисляем стандартное отклонение
std = train_data.std(axis=0)
# и делим на него
train_data /= std
# для подготовки тестовых даных используем среднее знач-е и станд.откл-е, вычисленное на обучающих данных
test_data -= mean
test_data /= std

### Построение сети

from keras import models
from keras import layers
#from tensorflow.keras.layers import Dropout

# ф-ция построения модели
def build_model():
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(12, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(12, activation='relu'))
    # для последнего слоя не указана ф-ция активации, чтобы сеть могла выдавать вещественные значения
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', 
                  loss='mse',       # ф-ция потерь - среднеквадратичная ошибка
                  metrics=['mae'])  # метрика - средняя абсолютная ошибка
    return model



model = build_model()
print("model.summary" , model.summary())


# ## Валидация модели с помощью механизма перекрёстной проверки по k блокам


import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences

# #оптимизация, борьба с переобучением 
# smaller_model = models.Sequential()
# smaller_model.add(layers.Dense(4, activation='relu',
#                            input_shape=(train_data.shape[1],)))
# smaller_model.add(layers.Dense(4, activation='relu'))
# smaller_model.add(layers.Dense(1))

# smaller_model.compile(optimizer='rmsprop', 
#                   loss='mse',       # ф-ция потерь - среднеквадратичная ошибка
#                   metrics=['mae'])
# smaller_model.summary()
# # Get a fresh, compiled small model.
# smaller_model_hist = smaller_model.fit(train_data, train_targets,
#                                        epochs=100,
#                                        batch_size=16,
#                                        validation_data=(test_data, test_targets))

# Get a fresh, compiled original model.
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=100, batch_size=16, validation_data=(test_data, test_targets))
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)


# epochs = range(1, 81)
# original_val_loss = original_hist.history['val_loss']
# smaller_model_val_loss = smaller_model_hist.history['val_loss']

# import matplotlib.pyplot as plt

# %matplotlib inline

# # b+ is for "blue cross"
# plt.plot(epochs, original_val_loss, 'b+', label='Original model')
# # "bo" is for "blue dot"
# plt.plot(epochs, smaller_model_val_loss, 'bo', label='Smaller model')
# plt.xlabel('Epochs')
# plt.ylabel('Validation loss')
# plt.legend()

# plt.show()


print('mae ', test_mae_score)

py = model.predict(test_data)
#test_targets - py
print('result', py)



# %%
