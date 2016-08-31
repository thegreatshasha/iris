from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f = load_iris()
pca = PCA(n_components=3)
tfit = pca.fit_transform(f['data'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
col = ['r', 'limegreen', 'b']
colors = [col[f['target'][i]] for i in range(len(f['data']))]

plt.scatter(tfit[:, 0], tfit[:, 1], s=60, c=colors, zs=tfit[:, 2])
plt.show()
