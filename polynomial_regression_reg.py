#import packages
from utils import *
import matplotlib.pyplot as pyplot
import numpy as numpy
import math

trainNumber = 100
deg = 8
lamda = [0,0.01,0.1,1,10,100,1000]
fold = 10

[t,X] = loadData()
X_n = normalizeData(X)
t = normalizeData(t)

def linearRegression(X, t, lamda, degree):
	expand = degexpand(X, degree)
	if lamda != 0:
		ridge = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(expand.transpose(),expand)+lamda*numpy.eye(len(expand[1]))),expand.transpose()),t)
	else:
		ridge = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(expand.transpose(),expand)),expand.transpose()),t)
	trainingError = numpy.mean(numpy.power(numpy.dot(expand, ridge) - t, 2))
	return [ridge, trainingError]


def evaluate(X,t,w,degree):
	expand = degexpand(X,degree)
	estimate = numpy.dot(expand,w)
	if len(t) > 0:
		testError = numpy.mean(numpy.power(estimate-t,2))
	else:
		testError = []
	return [estimate,testError]

X_n = X_n[:,2]
X_n = X_n.reshape(len(X_n),1)
trainX = X_n[:trainNumber,:]
trainT = t[:trainNumber]

space = math.ceil(trainNumber/fold)
step = np.arange(1,trainNumber,space)

index = 0
error = []

for i in lamda:
	currError = 0
	for j in range(fold - 1):
		testData = np.arange(step[j],(step[j+1]-1))
		trainData = np.setdiff1d(np.arange(1,trainNumber+1),testData)
		currTrainX = trainX[trainData-1,:]
		currTrainT = trainT[trainData-1]
		currTestX = trainX[testData,:]
		currTestT = trainT[testData]

		[w, trainED] = linearRegression(currTrainX,currTrainT,i,deg)
		[e, testED] = evaluate(currTestX,currTestT,w,deg)
		currError = testED + currError
	error.append(currError/fold)
	index += 1

pyplot.rcParams['font.size']=15
pyplot.semilogx(lamda,error,'r-o')
pyplot.plot([lamda[0],lamda[len(lamda)-1]],[error[0],error[0]],'b-o')
pyplot.xlabel('Regulation Value')
pyplot.ylabel('Average Validation Set Error')
pyplot.title('Polynomial Regression')
pyplot.show()