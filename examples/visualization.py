import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as datasets
from sklearn.mixture import GaussianMixture
# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Fit a Gaussian mixture model
gmm= GaussianMixture(n_components=3, random_state=0)
# gmm = GMM(n_components=3)
gmm.fit(X)
# Predict the labels
y_pred = gmm.predict(X)
# Plot the predicted labels

pred = pd.read_csv('/Users/kimyingwong/GMM_EM/dataset/testing.csv')

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot for the first scatter plot
axs[0].scatter(X[:, 0], X[:, 1], c=y_pred)
axs[0].set_xlabel('Sepal length')
axs[0].set_ylabel('Sepal width')
axs[0].set_title('Gaussian Mixture Model from sklearn')

# Plot for the second scatter plot
axs[1].scatter(X[:, 0], X[:, 1], c=pred["labels"])
axs[1].set_xlabel('Sepal length')
axs[1].set_ylabel('Sepal width')
axs[1].set_title('Gaussian Mixture Model from scratch')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


