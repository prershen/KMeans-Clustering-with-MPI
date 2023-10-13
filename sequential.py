import numpy as np
from scipy.cluster.vq import kmeans, whiten
import time
t1=time.time()
obs = whiten(np.genfromtxt('kmeans-master/data.csv',
dtype=float, delimiter=','))
K = 10
nstart = 10000
np.random.seed(0) # for testing purposes
centroids, distortion = kmeans(obs, K, nstart) #mean
(non-squared) Euclidean distance
print('Best distortion for %d tries: %f' % (nstart,
distortion))
t2=time.time()-t1
print("Time: %lf"%(t2))
