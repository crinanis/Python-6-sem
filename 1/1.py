#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Ксения
#
# Created:     21.02.2023
# Copyright:   (c) Ксения 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import sys
import pandas as pds
import mglearn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import IPython
import sklearn

print("версия Python: {}".format(sys.version))
print("версия pandas: {}".format(pds.__version__))
print("версия matplotlib: {}".format(matplotlib.__version__))
print("версия NumPy: {}".format(np.__version__))
print("версия SciPy: {}".format(sp.__version__))
print("версия IPython: {}".format(IPython.__version__))
print("версия scikit-learn: {}".format(sklearn.__version__))
print("версия scikit-learn: {}".format(mglearn.__version__))

##################################################################

from sklearn.datasets import load_iris

iris_dataset = load_iris() #загружаем набор данных Iris

print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193] + "\n...") #ключ, который содержит краткое описание набора данных

print("Названия ответов: {}".format(iris_dataset['target_names'])) #массив строк, содержащий сорта цветов, которые мы хотим предсказать

print("Названия признаков: \n{}".format(iris_dataset['feature_names'])) #это список строк с описанием каждого признака

#данные записаны в массивах таргет и дата
print("Тип массива data: {}".format(type(iris_dataset['data'])))

#Строки в массиве data соответствуют цветам ириса, а столбцы представляют собой четыре признака, которые были измерены для каждого цветка:
print("Форама массива data: {}".format(iris_dataset['data'].shape))
print("Первые пять строк массива data:\n{}".format(iris_dataset['data'][:5]))

#Массив target содержит сорта уже измеренных цветов
print("Тип массива target: {}".format(type(iris_dataset['target'])))

#target представляет собой одномерный массив, по одному элементу для каждого цветка
print("Форма массива target: {}".format(iris_dataset['target'].shape))

#Сорта кодируются как целые числа от 0 до 2:
print("Ответы:\n{}".format(iris_dataset['target']))

from sklearn.model_selection import train_test_split #перемешивает набор данных и разбивает его на две части.
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0) #используем фиксированное стартовое значение в гсч, чтобы повторно воспроизвести полученный результат

print("форма массива X_train: {}".format(X_train.shape)) #Х-потому что данные - двумерный массив
print("форма массива y_train: {}".format(y_train.shape)) #у потому что целевая переменная - одномерный массив

print("форма массива X_test: {}".format(X_test.shape))
print("форма массива y_test: {}".format(y_test.shape))

#маркируем столбцы, используя строки в iris_dataset.feature_names
iris_dataframe = pds.DataFrame(X_train, columns=iris_dataset.feature_names) #сначала преобразовываем массив нампи в датафрейм

from pandas.plotting import scatter_matrix #функция для создания парных диаграмм рассеивания, по диагонали гистограммы каждого признака

#создаём матрицу рассеяния из датафрейм, цвет задаём с помощью y_train
grr = scatter_matrix(
    iris_dataframe, c=y_train,
    figsize=(15, 15),
    marker='o',
    hist_kwds={'bins': 20},
    s=60, alpha=.8,
    cmap=mglearn.cm3
)

plt.show()
