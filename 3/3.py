#-------------------------------------------------------------------------------
# Name:        module3
# Purpose:
#
# Author:      Ксения
#
# Created:     28.02.2023
# Copyright:   (c) Ксения 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import mglearn
import numpy as np

# Классификация и регрессия
# Обобщающая способность, переобучение и недообучение
# набор данных для двухклассовой классификации, который содержит два признака

X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y) # заносим значения первого и вторго признаков
plt.legend(["Класс 0", "Класс 1"], loc=4)
plt.xlabel("Первый признак")
plt.ylabel("Второй признак")
print("форма массива X: {}".format(X.shape))
plt.show()

# синтетический набор для иллюстации алгоритма регрессии,
# содержит один входной признак и одну непрерывную переменную (отклик)
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Признак")
plt.ylabel("Целевая переменная")
plt.show()



# набор данных по раку молочной железы
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("Ключи cancer(): \n{}".format(cancer.keys()))

print("Форма массива data для набора cancer: {}".format(cancer.data.shape))

print("Количество примеров для каждого класса:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("Имена признаков:\n{}".format(cancer.feature_names))



from sklearn.datasets import load_boston
boston = load_boston()
print("форма массива data длянабора boston: {}".format(boston.data.shape))

# Набор данных содержит
# 506 точек данных и 13 признаков

# Конструирование признаков
# полученные 104 признака –13 исходных признаков плюс 91 производный признак.
X, y = mglearn.datasets.load_extended_boston()
print("формамассива X: {}".format(X.shape))


######################## Классификация с помощью k ближайших соседей

mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

# Теперь давайте посмотрим, как можно применить алгоритм k ближайших соседей, используя scikit-learn.
# раздел им наши данные на обучающий и тестовый наборы, чтобы оценить обобщающую способность модели
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

# заносим обучающий набор
clf.fit(X_train, y_train)

print("Прогнозы на тестовом наборе: {}".format(clf.predict(X_test)))

print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test, y_test)))




# как меняется граница принятия решений в зав-ти от кол-ва соседей

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # создаем объект-классификатор и подгоняем в одной строке
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("количество соседей:{}".format(n_neighbors))
    ax.set_xlabel("признак 0")
    ax.set_ylabel("признак 1")
axes[0].legend(loc=3)
plt.show()


# выясним, существует ли взаимосвязь между сложностью модели и обобщающей способностью
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # строим модель
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # записываем правильность на обучающем наборе
    training_accuracy.append(clf.score(X_train, y_train))
    # записываем правильность на тестовом наборе
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="правильность на обучающемн аборе")
plt.plot(neighbors_settings, test_accuracy, label="правильность на тестовом наборе")
plt.ylabel("Правильность")
plt.xlabel("количество соседей")
plt.legend()
plt.show()


################# Регрессия k ближайших соседей

mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()


# Существует также регрессионный вариант алгоритма k ближайших соседей.
# Опять же, давайте начнем с рассмотрения одного ближайшего соседа, на этот раз воспользуемся набором данных wave.
from sklearn.neighbors import KNeighborsRegressor
X, y = mglearn.datasets.make_wave(n_samples=40)
# разбиваем набор данных wave на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# создаем экземпляр модели и устанавливаем количество соседей равным 3
reg = KNeighborsRegressor(n_neighbors=3)
# подгоняем модель с использованием обучающих данных
reg.fit(X_train, y_train)

print("Прогнозы для тестового набора:\n{}".format(reg.predict(X_test)))

print("R^2 на тестовом наборе: {:.2f}".format(reg.score(X_test, y_test)))

####################### Анализ модели KNeighborsRegressor
# Применительно  к  нашему  одномерному  массиву  данных  мы  можем  увидеть прогнозы  для  всех  возможных  значений  признаков.
# Для  этого  мы  создаем тестовый набор данных и визуализируем полученные линии прогнозов:
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)

for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title(
    "{} neighbor(s)\ntrain score: {:.2f} test score: {:.2f}".format(
    n_neighbors, reg.score(X_train, y_train),
                 reg.score(X_test, y_test))
                 )
    ax.set_xlabel("Признак")
    ax.set_ylabel("Целеваяпеременная")
axes[0].legend(["Прогнозымодели", "Обучающиеданные/ответы", "Тестовыеданные/ответы"], loc="best")
plt.show()

# Как  видно  на  графике,  при  использовании  лишь  одного  соседа  каждая  точка обучающего набора имеет очевидное влияние на прогнозы,
# и предсказанные значения проходят  через  все  точки  данных.  Это  приводит  к  очень  неустойчивым  прогнозам.
# Увеличение числа соседей приводит к получению более сглаженных прогнозов, но при этом снижается правильность подгонки к обучающим данным.
