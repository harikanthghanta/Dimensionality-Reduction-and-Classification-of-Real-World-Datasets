import csv
import random as rd

def readData(file):
	data = list(csv.reader(open(file,'rb')))
	for i in range(0,len(data)):
		data[i] = [float(x) for x in data[i]]
	return data


def splitData(splitRatio):
	data = readData('pima-indians-diabetes.data1.csv')
	trainSetSize = int(splitRatio * len(data))
	trainSet = []
	while len(trainSet) < trainSetSize:
		i = rd.randrange(len(data))
		trainSet.append(data.pop(i))
	return [trainSet, data]

train , test = splitData(0.67)


def separateByClass(dataset):
	separate = {}
	#separate = {0.0:[],1.0:[]}
	for each in dataset:
		vector = each
		if(vector[-1] not in separate):
			separate[vector[-1]] = []
		separate[vector[-1]].append(vector)
	return separate


