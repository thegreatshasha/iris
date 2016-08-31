from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

f = load_iris()
pca = PCA(n_components=2)
tfit = pca.fit_transform(f['data'])

fig, ax = plt.subplots(1)
colors = [col[f['target'][i]] for i in range(len(f['data']))]
plt.scatter(tfit[:, 0], tfit[:, 1], c=colors)
plt.show()
