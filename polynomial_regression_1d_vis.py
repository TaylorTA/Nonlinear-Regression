#import packages
from utils import *
import matplotlib.pyplot as pyplot
import numpy as numpy

trainNumber = 100
degree = np.arange(1,11)

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
testX = X_n[trainNumber:,:]
testT = t[trainNumber:]

trainE = np.zeros(len(degree))
testE = np.zeros(len(degree))

for i in degree:
	[w, trainED] = linearRegression(trainX, trainT, 0, i)
	[e, testED] = evaluate(testX, testT, w, i)
	trainE[i-1] = trainED
	testE[i-1] = testED

	# plot a curve showing learned function
	x_ev = np.arange(min(X_n), max(X_n) + 0.1, 0.1).transpose()
	x_ev = x_ev.reshape(len(x_ev),1) 
	[y_ev,a] = evaluate(x_ev,[],w,i)# put your regression estimate here

	pyplot.plot(x_ev, y_ev, 'r.-')
	pyplot.plot(trainX, trainT, 'gx', markersize=10)
	pyplot.plot(testX, testT, 'bo', markersize=10, mfc='none')
	pyplot.xlabel('x')
	pyplot.ylabel('t')
	pyplot.title('Fig degree %d polynomial' % i)
	pyplot.show()