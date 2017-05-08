import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import naiveBayes as nb


#Load data set
data = pd.read_csv('temp1.csv', header=None)
df1 = data.iloc[:,-1]
del data[14]

#convert it to numpy arrays
X=data.values

pca = PCA(n_components=13)

pca.fit(X)

X1=pca.fit_transform(X)

df = pd.DataFrame(X1)

df[len(df.columns)] = df1

print df

train, test = nb.splitData1(df.values.tolist())
summaries = nb.summarizeByClass(train)
predictions = nb.getPredictions(summaries, test)
print nb.getAccuracy(test,predictions)