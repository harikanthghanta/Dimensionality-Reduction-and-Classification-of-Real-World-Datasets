import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

def scatterplot3d(class1_sample, class2_sample):
	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(111, projection='3d')
	plt.rcParams['legend.fontsize'] = 10   
	ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
	ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')

	plt.title('Samples for class 1 and class 2')
	ax.legend(loc='upper right')

	plt.show()

mean = [1, 10, 100]
cov = [[1,0.2,5], [0,1,0], [1,1,1]]
data = np.random.multivariate_normal(mean, cov, 1000)
class1_sample = data.T
df = pd.DataFrame(data)
df[3] = 0.0

#mean = [0.2, 3, 10]
mean = [0.9, 9, 99]
cov = [[1,0,0], [0,1,0], [1,1,1]]
#x, y, z = np.random.multivariate_normal(mean, cov, 1000).T

data1 = np.random.multivariate_normal(mean, cov, 1000)
class2_sample = data1.T
df1 = pd.DataFrame(data1)
df1[3] = 1.0
df = df.append(df1, ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('generateData.csv',index=None,header=None)
df.to_csv('../Decission tree/generateData.csv',index=None,header=None)

scatterplot3d(class1_sample, class2_sample)

