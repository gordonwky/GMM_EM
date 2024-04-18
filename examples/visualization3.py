import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pred = pd.read_csv('/Users/kimyingwong/GMM_EM/dataset/two_gaussian_predict_label.csv')
X = pd.read_csv('/Users/kimyingwong/GMM_EM/dataset/two_gaussian.csv', header=None)
print(X)

# Plot for the first scatter plot
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=pred["labels"])

plt.xlabel('x')
plt.ylabel('y ')
plt.title('Two Gaussian Mixture predicted from GMM-EM')

plt.show()
# Plot for the second scatter plot
# axs[1].scatter(X[:, 0], X[:, 1], c=pred["labels"])
# axs[1].set_xlabel('x')
# axs[1].set_ylabel('y')
# axs[1].set_title('Gaussian Mixture Model from scratch')
