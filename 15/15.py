import mglearn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_lfw_people

print(matplotlib.get_backend())
## ФАКТОРИЗАЦИЯ НЕОТРИЦАТЕЛЬНЫХ МАТРИЦ
mglearn.plots.plot_nmf_illustration()
plt.show()


people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
mask = np.zeros(people.target.shape, dtype=bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.0

X_train, X_test, y_train, y_test = train_test_split(X_people,
                                                    y_people,
                                                    stratify=y_people,
                                                    random_state=0)

#mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)
#plt.show()



nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)
fix, axes = plt.subplots(3,
                         5,
                         figsize=(15, 12),
                         subplot_kw={
                             "xticks": (),
                             "yticks": ()
                         })
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title(f"{i}. component")
plt.show()


compn = 3
# сортируем по 3-й компоненте,выводим первые 10 изображений
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))
plt.show()

compn = 7
# сортируем по 7-й компоненте,выводим первые 10 изображений
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))
plt.show()


S = mglearn.datasets.make_signals()
plt.figure(figsize=(6, 1))
plt.plot(S, "-")
plt.xlabel("Время")
plt.ylabel("Сигнал")
plt.show()

## 100 измерительных приборов
A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print(f"Форма измерений: {X.shape}")

# с помощью nmf восстанавливаем 3 сигнала
S_ = NMF(n_components=3, random_state=42, init=None,
         max_iter=10000).fit_transform(X)
print(f"Форма восстановленного сигнала: {S_.shape}")


# для сравнения ПСА
from sklearn.decomposition import PCA

H = PCA(n_components=3).fit_transform(X)

models = [X, S, S_, H]
names = [
    "Наблюдения (первые три измерения)",
    "Фактические источники",
    "Сигналы, восстановленные NMF",
    "Сигналы, восстановленные PCA",
]
fig, axes = plt.subplots(
    4,
    figsize=(8, 4),
    gridspec_kw={"hspace": 0.5},
    subplot_kw={
        "xticks": (),
        "yticks": ()
    },
)
for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], "-")
plt.show()


## алгоритмы множественного обучения t-SNE
from sklearn.datasets import load_digits # набор рукописных цифр

digits = load_digits()
fig, axes = plt.subplots(2,
                         5,
                         figsize=(10, 5),
                         subplot_kw={
                             "xticks": (),
                             "yticks": ()
                         })
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)
plt.show()


# строим модель PCA для визуализации данных, сведя их к двум измерениям.
pca = PCA(n_components=2).fit(digits.data)
# преобразуем данные рукописных цифр к первым двум компонентам
digits_pca = pca.transform(digits.data)
colors = [
    "#476A2A", "#7851B8", "#BD3430",
    "#4A2D4E", "#875525", "#A83683",
    "#4E655E", "#853541", "#3A3120", "#535D8E",
]
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits.data)):
    plt.text(
        digits_pca[i, 0],
        digits_pca[i, 1],
        str(digits.target[i]),
        color=colors[digits.target[i]],
        fontdict={
            "weight": "bold",
            "size": 9
        },
    )
plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")
plt.show()



from sklearn.manifold import TSNE
#в классе TSNE нет метода transform. Вместо этого мы можем вызвать метод fit_transform, который построит модель и немедленно вернет преобразованные данные
digits_tsne = TSNE(random_state=42, init="random",
                   learning_rate=200.0).fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    plt.text(
        digits_tsne[i, 0],
        digits_tsne[i, 1],
        str(digits.target[i]),
        color=colors[digits.target[i]],
        fontdict={
            "weight": "bold",
            "size": 9
        },
    )

plt.xlabel("t-SNE признак 0")
plt.ylabel("t-SNE признак 1")
plt.show()