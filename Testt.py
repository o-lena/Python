
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
from sklearn.metrics import r2_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

data = pd.read_csv('merc.csv')


nnpDf = data.sort_values("price",ascending = False).iloc[131:]

nnpDf.drop('transmission', axis=1, inplace=True)

nnpDf.drop('fuelType', axis=1, inplace=True)

nnpDf.drop('model', axis=1,inplace=True)



df = nnpDf


y = df[['price']].values
x = df.drop('price', axis=1).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 0)

#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# ####  Linear Regression ####

# from sklearn.linear_model import LinearRegression
# linear_regression = LinearRegression()
# linear_regression.fit(x_train,y_train)
# y_pred_lin_reg = linear_regression.predict(x_test)

# # print(f'y_test: {y_test}\nprediction: {y_pred_lin_reg}')

# # print('R Square Score for Linear Regression : ', r2_score(y_test, y_pred_lin_reg))

# ####  Polynomial Regression ####

# from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree=4)
# x_poly = poly_reg.fit_transform(x,y)
# linear_regression2 = LinearRegression()
# linear_regression2.fit(x_poly, y)
# y_pred_poly_reg = linear_regression2.predict(x_poly)

# # print(f'y: {y}\nprediction: {y_pred_poly_reg}')

# # print('R Square Score for Polynomial Regression : ', r2_score(y, y_pred_poly_reg))

# ####  SVR ####

# from sklearn.preprocessing import StandardScaler
# standard_scaler1 =StandardScaler()
# scaled_x_train = standard_scaler1.fit_transform(x_train)
# scaled_x_test = standard_scaler1.transform(x_test) # dont learn just transform
# standard_scaler2=StandardScaler()
# scaled_y_train = np.ravel(standard_scaler2.fit_transform(y_train.reshape(-1,1)))
# scaled_y_test = np.ravel(standard_scaler2.fit_transform(y_test.reshape(-1,1)))

# from sklearn.svm import SVR
# svr_regression = SVR(kernel="rbf") # other kernel types : linear, poly, rbf, sigmoid
# svr_regression.fit(scaled_x_train, scaled_y_train)
# y_pred_svr_reg = svr_regression.predict(scaled_x_test)

# # print(f'y_scaled: {scaled_y_test}\nprediction: {y_pred_svr_reg}')

# # print('R Square Score for Support Vector Regression : ', r2_score(scaled_y_test, y_pred_svr_reg))

# #### Decision Tree Regression ####

# from sklearn.tree import DecisionTreeRegressor
# decision_tree_reg = DecisionTreeRegressor(random_state=0)
# decision_tree_reg.fit(x_train, y_train)
# y_pred_dt_reg = decision_tree_reg.predict(x_test)

# # print(f'y_test: {y_test}\nprediction: {y_pred_dt_reg}')

# # print('R Square Score for Decision Tree Regression : ', r2_score(y_test, y_pred_dt_reg))


####  Random Forest Regression ####

from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor(n_estimators = 50, random_state= 0 ) # n_estimators = numbor of estimator tree
random_forest_reg.fit(x_train, y_train.flatten())
y_pred_rf_reg = random_forest_reg.predict(x_test)

print(f'y_test: {y_test}\nprediction: {y_pred_rf_reg}')


print('R Square Score for Random Forest Regression : ', r2_score(y_test, y_pred_rf_reg))

def prediction(rows):
     #variable=pd.read_csv(data).values
     prediction=random_forest_reg.predict(rows)
     #print('prdct', prediction)
     #list1 = prediction.tolist()
     return prediction

print(random_forest_reg.predict([[2000, 12000, 150, 31.4, 3.0]]))



######  VBA-Versuch ######

# import xlwings as xw
# import sys

# #filePath = r"C:\Users\Lena\Desktop\python\Mappe1.xlsm"
# filePath = sys.argv[1]
# #params = 'A1:A6'
# #params = sys.argv[2]

# wb = xw.Book(filePath)
# sheet = wb.sheets['Tabelle1']

# rows = sheet.range('A1:E4').options(ndim=2).value

# variable = prediction(rows)
# for var in variable:
#     print(var)

# # sheet.range('G1').value = np.expand_dims(variable, axis=1).tolist()

# # wb.save()







