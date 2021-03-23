#!/usr/bin/env python
# coding: utf-8

#%%
import keras
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#dataset = df.values

from keras import models
from keras import layers
from keras.optimizers import SGD
# ## Набор данных

data = pd.read_csv('merc.csv')

nnpDf = data.sort_values("price",ascending = False).iloc[131:]

nnpDf.drop(['transmission', 'fuelType', 'model'], axis=1, inplace=True)

df = nnpDf

df.info()

y = df[['price']].values
x = df.drop('price', axis=1).values

from sklearn.model_selection import train_test_split
train_data, test_data, train_targets, test_targets = train_test_split(x, y, test_size = 0.2, random_state = 0)

print(train_data.shape,train_targets.shape,test_data.shape,test_targets.shape)


# from sklearn.preprocessing import MinMaxScaler
# mms = MinMaxScaler()
# test_data = mms.fit_transform(train_data)
# test_targets= mms.transform(test_data)

#вычисление среднего значения по всем обучающим данным

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
test_data = scaler.fit_transform(test_data)
#test_targets=scaler.fit_transform(test_targets)

train_data=scaler.fit_transform(train_data)
#train_targets=scaler.fit_transform(train_targets)

# test_data = StandardScaler().fit_transform(test_data)
# test_targets= StandardScaler().fit_transform(test_targets.reshape(len(test_targets),1))[:,0]

# train_data=StandardScaler().fit_transform(train_data)

# ## Построение сети



# ф-ция построения модели
def build_model():

    model = models.Sequential()
    model.add(layers.Dense(1, activation='relu',
                        kernel_initializer='he_uniform',
                           input_shape=(train_data.shape[1],)))
    # model.add(layers.Dropout(0.8))
    
    #model.add(layers.Dense(7, activation='relu'))
    # model.add(layers.Dropout(0.8))
    # для последнего слоя не указана ф-ция активации, чтобы сеть могла выдавать вещественные значения
    #model.add(layers.Dense(1, activation='linear'))

    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='mean_absolute_error', 
                  optimizer=opt,
                  metrics=['mse'])

    # model.add(layers.Activation('softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['mae'])
    # model.compile(optimizer='rmsprop', 
    #               loss='mse',       # ф-ция потерь - среднеквадратичная ошибка
    #               metrics=['mae'])  # метрика - средняя абсолютная ошибка
    return model


model = build_model()
print("model.summary" , model.summary())


# ## Валидация модели с помощью механизма перекрёстной проверки по k блокам


import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences

#padded = pad_sequences(train_data)
#x_train_data = np.expand_dims(padded, axis = 0)

#y = np.array([1,0,1])
#x_train_targets = train_targets.reshape(1,-1)

# x_train_data = np.array(train_data) # problem numpy
# x_train_targets=np.array(train_targets)
# print(len(x_train_data))


# Get a fresh, compiled model.
model = build_model()
# Train it on the entirety of the data.
history=model.fit(train_data, train_targets,
          epochs=100,  batch_size=1000, verbose=0)
#test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
train_mse = model.evaluate(test_data, test_targets, verbose=0)
test_mse = model.evaluate(test_data, test_targets, verbose=0)

print('Train',(train_mse, test_mse))

# # plot loss during training
# plt.subplot(211)
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# # plot mse during training
# plt.subplot(212)
# plt.title('Mean Squared Error')
# plt.plot(history.history['mean_squared_error'], label='train')
# plt.plot(history.history['val_mean_squared_error'], label='test')
# plt.legend()
# plt.show()


#print('err', test_mae_score)

py = model.predict([[2000, 12000, 150, 31.4, 3.0]])
#test_targets - py
print('predict', py)



# %%
