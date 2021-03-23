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
## for gui 
def show_data(xlsx_path, model):
        df = pd.read_excel(xlsx_path, engine='openpyxl')
        #df = pd.read_excel(r"C:\Users\Lena\Desktop\python\merc.xlsx", engine='openpyxl')
        df.drop(['price'], axis=1,inplace = True)
        df_a = df[df["model"] == model]
        # df_b = df_a[df_a['year']==year]
        # df_c = df_b[df_b['mileage']==mileage]
        x = []
        for row in df_a.values:
            num_list = row.tolist()
            x.append(num_list)
        return(x)

#print(show_data('A Class', 2020, 606))
def prediction_data(sort):
    list1=[]
    list2 = []
    list1.append(sort)
    for item in list1:
        item.pop(0)
        item.pop(1)
        item.pop(2)
        list2.append(item)
    return (list2)

#print(prediction_data(r"C:\Users\Lena\Desktop\python\merc.xlsx","A Class", 1.5))




def RandomForestRegression(predict):
    data = pd.read_csv('merc.csv')
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
    prediction = prediction_data(predict)
    #print(f'y_test: {y_test}\nprediction: {y_pred_rf_reg}')
    #print('R Square Score for Random Forest Regression : ', r2_score(y_test, y_pred_rf_reg))
    return ('estimated price: ', random_forest_reg.predict(prediction))

### for prediction 

# def get_data(xlsx_path,model):
#     data = show_data()
#     for item in data:

def get_data(xlsx_path, model, year, mileage):
    df = pd.read_excel(xlsx_path, engine='openpyxl')
    #df = pd.read_excel(r"C:\Users\Lena\Desktop\python\merc.xlsx", engine='openpyxl')
    df.drop(['transmission','fuelType', 'price'], axis=1,inplace = True)
    df_a = df[df["model"] == model]
    df_b = df_a[df_a['year']==year]
    df_c = df_b[df_b['mileage']==mileage]
    x = []
    for row in df_c.values:
        num_list = row.tolist()
        x.append(num_list)
    list1=[]
    for item in x:
        list1.append(item[1:5])
    return (list1)
    #return(x)
print(get_data(r"C:\Users\Lena\Desktop\python\merc.xlsx","A Class", 2020, 606))

def get_prediction_df(xlsx_path,model, year, mileage):
    #data=get_data(r"C:\Users\Lena\Desktop\python\merc.xlsx","A Class")
    #print(type(data))
    data =get_data(xlsx_path, model, year, mileage)
    list1=[]
    for item in data:
        list1.append(item[1:5])
    return (list1)
print(get_prediction_df(r"C:\Users\Lena\Desktop\python\merc.xlsx","A Class", 2020, 606))




##### Interface creation

import json

#Konfigurationsdatei
with open("data_file.json") as json_data_file:
    jdata = json.load(json_data_file)
#Zugriff Konfigurationsdatei
tab4=jdata["tab4"]

#sg.theme_previewer() #show all themes
sg.theme('Default') #choose a theme    
sg.SetOptions(element_padding=(10, 10))      

# Menu definition     
menu_def = [['HELP', 'How to use the application']]  
item = []
values_t = []
header = ['year price transmission mileage fuelType tax mpg engineSize']
# Layout creation 
layout = [ [sg.Menu(menu_def, )],
                                    [sg.Text('Price forecasting application',font=("Helvetica", 15) )],
           [sg.T('Enter xlsx file path', size=(15, 1), font=("Helvetica", 15))],
           [sg.InputText(key='Input_XLSX_path'), sg.FileBrowse()],
           [sg.T('Enter model name',size=(15, 1), font=("Helvetica", 15))], 
           [sg.InputText(key='Input_model', size=(15, 1)), sg.B('Confirm', key='ConfirmPath')], 
           [sg.T('Dataset', size=(33, 1), font=("Helvetica", 15)) , sg.T('Prediction output', size=(15, 1), font=("Helvetica", 15))],
           [sg.Listbox(item,size=(50, 10),key='listbox', enable_events=True), sg.Listbox(values_t,size=(50, 10),key='listbox2', enable_events=True)],
           [sg.B('Predict'),]]      

# layout2 = [[sg.Text('Choose a predictio model')], 
#         [sg.Drop(values = tab4["predict_model"], key='model', size=(10,10))], 
#         [sg.B('Show result')],
#         [sg.Listbox(values_t,size=(100, 20),key='listbox2', enable_events=True)],
#         [sg.B('Export to Excel'),sg.B('Clean'),sg.B('Infobox')]]

window = sg.Window('CSV to Nextcloud parser', layout)
# window2 = sg.Window("Test Anatomy - Main Menu", layout2)

# Loop to choose buttons  
while True:      
    event, values = window.read()   
    xlsxPath = values['Input_XLSX_path'] 
    model = values['Input_model'] 
    if event == 'ConfirmPath':
        window.FindElement('listbox').Update('')
        item = show_data(xlsxPath, model)
        window.FindElement('listbox').Update(header+item)
    if event == 'Predict': 
        get = values['listbox']
        do = get[0]
        #prediction_data(do)
        window.FindElement('listbox2').Update('')
        values_t=RandomForestRegression(do)
        window.FindElement('listbox2').Update(values_t)
        #window2 = sg.Window("Test Anatomy - Main Menu", layout2)
        # event, values = window2.read()
        # if event == 'Show result':
        #     window2.FindElement('listbox2').Update('')
        #     values=RandomForestRegressor()
        #     window2.FindElement('listbox2').Update(values)

        #liste aktualisieren
        #choices = search_data(textInputs_such)
        #window.FindElement('listbox').Update(item)  

    if event == sg.WIN_CLOSED or event == 'Exit':      
        break            

    # Process menu choices  
    if event == 'How to use the application':      
        sg.popup("Step 1: For the application to work, click 'Browse' and select the requireed file",  # popup is a GUI equivalent of a print statement
                 "Step 2: Click 'Confirm' if the path is correct",
                 "Step 3: Click  'Create isc file'",
                 "Step 4: Upload the created ics file to your Nextcloud calendar")      
  

		

window.close()
