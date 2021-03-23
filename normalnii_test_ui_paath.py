import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
# ------- IMPORTS -------
import PySimpleGUI as sg 
import csv
import pytz
import os
import csv
import time
import datetime
from datetime import date
import numpy as np #für mehrdimensionale arrays
import configparser
import json
import win32com.client as win32
import pandas as pd
import re
import numpy as np
import xlwings as xw
import sys

import numpy as np
import ast


def umwandlung(strings):
    var = strings
    li = list(var)
    lis = [s.split(',') for s in li]
    x = []
    for n in lis: 
        x.append([ast.literal_eval(i) for i in n])
    return(x)
       
#print(umwandlung({"2017,27000,20,61.4,2.1", "2016,6200,555,28,5.5", "2016,16000,325,30.4,4"}))

def RandomForestRegression(prediction,data):
    # filePath = r"C:\Users\Lena\Desktop\python\test_for_ui-path.xlsx"
    # wb = xw.Book(filePath)
    # sheet = wb.sheets['Tabelle1']
    # prediction=str(prediction)
    # rows = sheet.range(prediction).options(ndim=2).value
    data = pd.read_csv(data)
    nnpDf = data.sort_values("price",ascending = False).iloc[131:]
    nnpDf.drop('transmission', axis=1, inplace=True)
    nnpDf.drop('fuelType', axis=1, inplace=True)
    nnpDf.drop('model', axis=1,inplace=True)
    df = nnpDf  
    y = df[['price']].values
    x = df.drop('price', axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 0)
    random_forest_reg = RandomForestRegressor(n_estimators = 50, random_state= 0 ) # n_estimators = numbor of estimator tree
    random_forest_reg.fit(x_train, y_train.flatten())
    y_pred_rf_reg = random_forest_reg.predict(x_test)
    # sheet.range('F2').value = np.expand_dims(random_forest_reg.predict(rows), axis=1).tolist()
    # wb.save()
    #prediction = prediction_data(predict)
    #print(f'y_test: {y_test}\nprediction: {y_pred_rf_reg}')
    #print('R Square Score for Random Forest Regression : ', r2_score(y_test, y_pred_rf_reg))
    return (random_forest_reg.predict(umwandlung(prediction)))


#print(RandomForestRegression({"2017,27000,20,61.4,2.1", "2016,6200,555,28,5.5", "2016,16000,325,30.4,4"},'merc.csv'))