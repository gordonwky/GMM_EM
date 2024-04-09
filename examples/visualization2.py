import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as datasets
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()

# Load the iris dataset
breast = datasets.load_breast_cancer()
X = breast.data
y = breast.target
print(X)
# Fit a Gaussian mixture model
gmm= GaussianMixture(n_components=2, random_state=0)
# gmm = GMM(n_components=3)
gmm.fit(pd.DataFrame(scaler.fit_transform(X)))
# print(gmm.means_)
# Predict the labels
y_pred = gmm.predict(pd.DataFrame(scaler.fit_transform(X)))
# Plot the predicted labels

pred_csv = pd.read_csv('/Users/kimyingwong/GMM_EM/dataset/breast_labels.csv')
pred = np.array(pred_csv["labels"])

print(confusion_matrix(pred, y_pred))

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot for the first scatter plot
axs[0].scatter(X[:, 0], X[:, 1], c=y_pred)
axs[0].set_xlabel('radius_mean')
axs[0].set_ylabel('texture_mean')
axs[0].set_title('Gaussian Mixture Model from sklearn')

# Plot for the second scatter plot
axs[1].scatter(X[:, 0], X[:, 1], c=pred)
axs[1].set_xlabel('radius_mean')
axs[1].set_ylabel('texture_mean')
axs[1].set_title('Gaussian Mixture Model from scratch')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
# print(confusion_matrix(pred, y_pred))



