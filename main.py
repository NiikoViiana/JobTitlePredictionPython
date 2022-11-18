# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import sys

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import gc
from tqdm.notebook import tqdm
from mpl_toolkits.basemap import Basemap

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix, mean_absolute_error
from bayes_opt import BayesianOptimization
import scikitplot as skplt
import lightgbm as lgb

warnings.filterwarnings('ignore')

# EDA Exploratory Data Analysis
originalData = pd.read_csv("Dataset/jobss.csv")
originalData.head()
originalData.info()

# De la data original se identifica la cantidad de Features que existen
# con su tipo de dato, verificando el contexto del analisis, se determina que
# algunas features no son necesarias para la predicción del Job Title
# por lo que se realiza una manipulación de la data, para dejar solo el Job Title & las Key Skills
# tambien se realiza una agrupacion por Industry para sacar los Job Title mas relevantes
# a su vez tambien se identifica que existen datos nulos y con caracteres peligrosos
# por lo que se realiza una limpieza de datos

# Limpieza de los datos
cleanData = pd.read_csv("Dataset/cleanjobs.csv")

# print("\n Quick view of the data: \n")
cleanData.head()

cleanData.Key_Skills.info()

newSkills = []

for skills in cleanData.Key_Skills:
    arraySkills = skills.split('|')

    for skill in arraySkills:
        newSkills.append(skill)

idx = pd.Index(newSkills, name='Key_Skills')

# Se verifica que
idx.value_counts(normalize=True)
idx.value_counts(normalize=True).plot.pie()
plt.show()

# Gracias a una grafica, se puede observar la recurrencia de años que exigen
# Cada uo de los datos
cleanData.Job_Experience_Required.value_counts(normalize=True)
cleanData.Job_Experience_Required.value_counts(normalize=True).plot.pie()
plt.show()


# Entrenamiento del modelo
y = cleanData["ID"]
del cleanData["ID"]
gc.collect()

X_train, X_val, y_train, y_val = train_test_split(cleanData, y, test_size = 0.2, random_state=42)

def evaluateRegressor(true, predicted, message="Test set"):
    MSE = mean_squared_error(true, predicted, squared=True)
    MAE = mean_absolute_error(true, predicted)
    RMSE = mean_squared_error(true, predicted, squared=False)
    LogRMSE = mean_squared_error(np.log(true), np.log(predicted), squared=False)
    print(message)
    print("MSE:", MSE)
    print("MAE:", MAE)
    print("RMSE:", RMSE)
    print("LogRMSE:", LogRMSE)


def PlotPrediction(true, predicted, title="Dataset: "):
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(111)
    ax1.set_title(title + 'True vs Predicted')
    ax1.scatter(list(range(0, len(true))), true, s=10, c='r', marker="o", label='True')
    ax1.scatter(list(range(0, len(predicted))), predicted, s=10, c='b', marker="o", label='Predicted')
    plt.legend(loc='upper right');
    plt.show()

LGBMReg = lgb.LGBMRegressor().fit(X_train, y_train)
pred = LGBMReg.predict(X_val)

evaluateRegressor(y_val, pred, "Train Set:")
PlotPrediction(y_val, pred, "Train Set: ")