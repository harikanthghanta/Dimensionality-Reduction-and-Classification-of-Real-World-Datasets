import csv
import math
import random as rd

def readData(file):
	data = list(csv.reader(open(file,'rb')))
	for i in range(0,len(data)):
		data[i] = [float(x) for x in data[i]]
	return data

#readData('pima-indians-diabetes.data1.csv')


def splitData(splitRatio):
	data = readData('pima-indians-diabetes.data1.csv')
	trainSetSize = int(splitRatio * len(data))
	trainSet = []
	while len(trainSet) < trainSetSize:
		i = rd.randrange(len(data))
		trainSet.append(data.pop(i))
	return [trainSet, data]



# print train
# print "**************************************************************"
# print test
#comiited

def separateByClass(dataset):
	separate = {}
	#separate = {0.0:[],1.0:[]}
	for each in dataset:
		vector = each
		if(vector[-1] not in separate):
			separate[vector[-1]] = []
		separate[vector[-1]].append(vector)
	return separate


def mean(var):
    return sum(var)/float(len(var))
    
    
def stdDev(var):
    m = mean(var)
    variance = sum([pow(x-m,2) for x in var])/ float(len(var))
    return math.sqrt(variance)
    
# print zip(*train)  adds 1st element of lists in

def summarize(data):
    summaries = [(mean(attr), stdDev(attr)) for attr in zip(*data)]
    del summaries[-1]
    return summaries
        
def summarizeByClass(data):
    separated = separateByClass(data)
    summaries = {}
    for classValue, instances in separated.iteritems():    
        summaries[classValue] = summarize(instances)
    return summaries
    
def calProb(x, mean, stddev):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stddev, 2))))
    return (1/(math.sqrt(2*math.pi)*stddev))*exponent

def calculateClassProb(summaries, inputData):
	prob = {}
	for classValue, classSummaries in summaries.iteritems():
		prob[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stddev = classSummaries[i]
			x = inputData[i]
			prob[classValue] *= calProb(x, mean, stddev)
	return prob

def predict(summaries, inputData):
	probs = calculateClassProb(summaries, inputData)
	className, bestProb = None, -1
	for classValue, prob in probs.iteritems():
		if className is None or prob > bestProb :
			bestProb = prob
			className = classValue
	return className

def getPredictions(summaries, testData):
	predictions = []
	for each in testData:
		res = predict(summaries, each)
		predictions.append(res)

	return predictions
def getAccuracy(testData, predictions):
	correct = 0 
	for x in range(len(testData)):
		if testData[x][-1] == predictions[x]:
			correct += 1
	return 100.0*correct/float(len(testData))
			

def main():
	print "in main"
	train, test = splitData(0.67)
	summaries = summarizeByClass(train)
	predictions = getPredictions(summaries, test)
	print getAccuracy(test,predictions)

main()



			



    
    
    
    
    
    
    
    
    
    
    
    
    
    




