import numpy as np
from scipy.cluster.vq import kmeans, whiten
from operator import itemgetter
from math import ceil
from mpi4py import MPI
import time
comm = MPI.COMM_WORLD
rank = comm.Get_rank(); size = comm.Get_size()
np.random.seed(seed=rank)
t1=time.time()
obs = whiten(np.genfromtxt('kmeans-master/data.csv',
dtype=float, delimiter=','))
K = 10; nstart = 10000
n = int(ceil(float(nstart) / size))
centroids, distortion = kmeans(obs, K, n)
results = comm.gather((centroids, distortion), root=0)
if rank == 0:
results.sort(key=itemgetter(1))
result = results[0]
print('Best distortion for %d tries: %f' %
(nstart, result[1]))
t2=time.time()-t1
print("Time: %lf"%(t2))
