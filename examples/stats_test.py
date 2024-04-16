import numpy as np
from scipy.stats import norm
data = [1,2,3]
m,s = norm.fit(data)
print(norm.pdf(data,m,s))

from sklearn.mixture import GaussianMixture
# import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
import time

# gmm = GaussianMixture(n_components=3, random_state=0)
# gmm.fit(data)


# log_likelihood = np.log(np.product(norm.pdf(data,m,s)))
# print(log_likelihood)
mean1 = np.array([0, 0])

cov1 = np.array([[1, 0.5],
                [0.5, 1]])

samples1 = np.random.multivariate_normal(mean1, cov1, size = 50)

# # mean2 = np.array([15, 20])

# cov2 = np.array([[1, 0.5],
#                 [0.5, 1]])
# samples2 =np.random.multivariate_normal(mean2, cov2, size = 100000)


mean3 = np.array([5, 10])

cov3 = np.array([[1, 0.5],
                [0.5, 1]])
samples3 =np.random.multivariate_normal(mean3, cov3, size = 50)

sample = np.concatenate((samples1,samples3), axis=0)


# print(multivariate_normal.pdf(sample, mean=[4.44322,  9.99744], cov=np.identity(2)))

pd.DataFrame(sample).to_csv('/Users/kimyingwong/GMM_EM/dataset/two_gaussian.csv', header=False, index=False)
# gmm = GaussianMixture(n_components=2, random_state=0)
# # time_start = time.time()
# gmm.fit(sample)
# ypred = gmm.predict(sample)
# print(time.time()-time_start)
# print(gmm.score(sample))


# Plot for the first scatter plot
# plt.scatter(sample[:, 0], sample[:, 1], c = ypred)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Datapoints from two Gaussian Distribution')
# plt.show()
# print(sample)

# print(gmm.covariances_) # Covariances
# print(gmm.means_) # Means

# print("Generated samples 1:")
# print(samples1)

# print("Generated samples 2:")
# print(samples2)