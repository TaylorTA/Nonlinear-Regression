#import packages
from utils import *
import matplotlib.pyplot as pyplot
import numpy as numpy

trainNumber = 100
degree = np.arange(1,11)

[t,X] = loadData()
X_n = normalizeData(X)
t = normalizeData(t)
	
trainX = X_n[:trainNumber,:]
trainT = t[:trainNumber]
testX = X_n[trainNumber:,:]
testT = t[trainNumber:]

trainE = np.zeros(len(degree))
testE = np.zeros(len(degree))


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

# calculate errors in different polynomial degree
for i in degree:
	[w, trainED] = linearRegression(trainX, trainT, 0, i)
	[e, testED] = evaluate(testX, testT, w, i)
	trainE[i-1] = trainED
	testE[i-1] = testED

#print the plot of errors versus polynomial degree
pyplot.rcParams['font.size']=15
pyplot.xlabel('Polynomial Degree')
pyplot.ylabel('Error (mean squared error)')
pyplot.legend(['Train Error','Test Error'])
pyplot.plot(degree,trainE, 'b-o')
pyplot.plot(degree,testE, 'r-o')
pyplot.title('Errors in Different Polynomial Degree')
pyplot.show()




