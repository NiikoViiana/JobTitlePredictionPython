# For Importing data {Kaggle.}
import os
for dirname, _, filenames in os.walk('Dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import regex as re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#For Data Viz.
import matplotlib.pyplot as plt
import seaborn as sns

#For Warnings
import warnings
warnings.filterwarnings('ignore')

#For Options
pd.set_option('display.max_columns' , None)
pd.set_option('display.max_rows', None)

#From Sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# For NLP related task

from PIL import Image
import requests

# Leyendo el DataSet
filepath = "Dataset/cleanjobs.csv"
missing_values = ["NaN", " ", "--", "na", "Not Mentioned"]
data_import = pd.read_csv(filepath , na_values = missing_values)
data_import.head()

# EDA Exploratory Data Analysis
# Se elimina la columna que no tiene una definición clara
# df = data_import.drop('Unnamed: 1', axis = 1)
df = data_import
df.head()
# Se copia el dataset sin la columna
df1 = df.copy()
# Se verifican los tipos de objetos de cada una de las columnas
df.info()

# Se verifican los datos vacios del DataSet
def missing_vals(df):
    for i in df:
        print(f"{i}:{df[i].isnull().sum()} de {len(df[i])} datos")

missing_vals(df)

# Se identifican las columnas que son de tipo categorico
def cat_cols(df) :
    o = (df.dtypes == 'object')
    object_cols = o[o].index
    return object_cols

object_cols = cat_cols(df)
print(f"Las columnas categoricas son : {object_cols}")

# Se identifican los posibles numeros que existan en las columnas categoricas
def detecting_num_in_cat_columns(df):
    cnt=0
    for row in df:
        if type(row) == 'object' :
            try:
                int(row)
                df.loc[cnt, row] = np.nan
            except ValueError:
                pass
            cnt+=1
    return f"Existen {cnt} numeros en las columnas categoricas."

print(detecting_num_in_cat_columns(df))

# Se toman los años de experiencia sin repetir
print(df['Job_Experience_Required'].unique())
# Se limpian los datos nulos y datos que no corresponden
df['Job_Experience_Required'] = df['Job_Experience_Required'].replace('vide', 0)
df['Job_Experience_Required'] = df['Job_Experience_Required'].replace(np.nan, 0)
print(df['Job_Experience_Required'].unique())
# Se verifica si existe algun dato nullo
print(f"Datos nulos: {df['Job_Experience_Required'].isnull().sum()}" )

# Detectando el total de la experiencia requerida
experience_required = []
numbers_list = []

def experience_required_func(df):
    for i in df['Job_Experience_Required']:
        if type(i) == str:
            if re.search(r'\d+', i):
                numbers = re.findall(r'\d+', i)
                numbers_list.append(numbers)
        else:
            numbers_list.append([0])

    for number in numbers_list:

        if len(number) != 1:
            first_num = int(number[0])
            second_num = int(number[1])
            sub = second_num - first_num
            experience_required.append(abs(sub))
        else:
            experience_required.append(0)

    return experience_required

experience_required = experience_required_func(df)

print(len(experience_required))

#Visualización de los datos
# Porcentaje de trabajos con más experiencia
data_viz_df = df.copy()
data_viz_df['Experiencia en años'] = experience_required
data_viz_df.head()
# Graficamente
data_viz_df['Experiencia en años'].value_counts()[:5].plot(kind = 'pie', autopct = '%1.1f%%', shadow = True, explode = [0.1, 0, 0, 0, 0])
plt.title('Porcentaje de trabajos con mas experiencia')
fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.show()
# Se puede concluir que casi el 34% de las personas tienen una experiencia de 5 años en el dataset
# También el 23,1% de las personas tiene una experiencia de 2 años en el dataset

# Visualización de los datos
# Tipos comunes de trabajos
data_viz_df['Tipos comunes de trabajos'] = data_viz_df['Job_Title']

def job_types(data):
    for i in data:
        if type(i) == str:
            if 'desarrollador' in i.lower():
                data_viz_df['Tipos comunes de trabajos'].replace(to_replace=i.lower(), value='Desarrolladores')

            elif 'disenador' in i.lower():
                data_viz_df['Tipos comunes de trabajos'].replace(to_replace=i.lower(), value='Diseñador')

            elif 'gerente' in i.lower():
                data_viz_df['Tipos comunes de trabajos'].replace(to_replace=i.lower(), value='Gerente')
        else:
            data_viz_df['Tipos comunes de trabajos'].replace(i, 'Otros')

    return data_viz_df['Tipos comunes de trabajos']

data_viz_df['Tipos comunes de trabajos'] = job_types(data_viz_df['Job_Title'])

common_jobs = []
for i in data_viz_df['Job_Title']:
    if type(i) == str:
        if 'desarrollador' in i.lower():
            common_jobs.append('Desarrolladores')
        elif 'disenador' in i.lower():
            common_jobs.append('Diseñador')
        elif 'gerente' in i.lower():
            common_jobs.append('Gerente')
        else:
            common_jobs.append('Otros')
    else :
        common_jobs.append('Otros')

data_viz_df['Tipos comunes de trabajos'] = common_jobs

# data_viz_df = data_viz_df.drop(['Role', 'Industry'] , axis = 1)

print("Conteo", data_viz_df['Tipos comunes de trabajos'].value_counts())

data_viz_df['Tipos comunes de trabajos'].value_counts().plot(kind = 'pie', autopct = '%1.1f%%', shadow = True, explode = [0.1, 0, 0, 0])
plt.title('Tipos comunes de trabajos')
fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.show()

# Visualización de resultados
print(data_viz_df[['Experiencia en años', 'Tipos comunes de trabajos', 'ID']]\
    .groupby(['Experiencia en años', 'Tipos comunes de trabajos'])\
    .mean().sort_values(by = 'Experiencia en años', ascending = False))
