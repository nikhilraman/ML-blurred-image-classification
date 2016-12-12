"""
=================================================================================
File for training an SVM on the small datasets and then gathering training/test stats 
=================================================================================

""" 

print(__doc__) 

#### IMPORTS #### 

import numpy as np 
from PIL import Image
import sys 
import os
import scipy 
from numpy import loadtxt
from scipy import ndimage 
from sklearn.metrics import accuracy_score  
from sklearn import svm 


if __name__ == "__main__": 

	# Main code  

	##################################
	##### Load the training data ##### 
	##################################

	print "Loading in Training Data"

	## Blur ##
	filename = 'data/blur-train-full.dat' 
	data = loadtxt(filename, delimiter=',') 
	Xtrain_blur = data 

	n,d  = Xtrain_blur.shape  
	print n
	ytrain_blur = np.ones(n) 

	## No Blur ##
	filename2 = 'data/no-blur-train-full.dat' 
	data2 = loadtxt(filename2, delimiter=',') 
	Xtrain_no_blur = data2 

	n2,d  = Xtrain_no_blur.shape  
	print n2
	ytrain_no_blur = np.zeros(n2) 

	# Aggregate the training data
	Xtrain = np.concatenate((Xtrain_blur, Xtrain_no_blur), axis=0)
	print Xtrain
	ytrain = np.append(ytrain_blur, ytrain_no_blur) 
	print ytrain


	#################################
	##### Load the testing data #####
	################################# 

	print "Loading in Testing Data"

	## Blur ##
	filename3 = 'data/blur-test-full.dat' 
	data3 = loadtxt(filename3, delimiter=',') 
	Xtest_blur = data3 

	n3,d  = Xtest_blur.shape 
	print n3
	ytest_blur = np.ones(n3) 

	## No Blur ##
	filename4 = 'data/no-blur-test-full.dat' 
	data4 = loadtxt(filename4, delimiter=',') 
	Xtest_no_blur = data4 

	n4,d  = Xtest_no_blur.shape 
	print n4
	ytest_no_blur = np.zeros(n4) 

	# Aggregate the testing data
	Xtest = np.concatenate((Xtest_blur, Xtest_no_blur), axis=0)
	ytest = np.append(ytest_blur, ytest_no_blur)

	##### Fit the model and output metrics ##### 

	print "Fitting the Gaussian SVM"

	svmModel = svm.SVC(kernel="rbf")

	svmModel.fit(Xtrain, ytrain)  

	print "Predicting Labels"

	ypreds = svmModel.predict(Xtest) 

	smallSampleAccuracyScore = accuracy_score(ypreds, ytest)

	print "100 train, 100 test -- TEST ACCURACY: " + str(smallSampleAccuracyScore) 

	ypreds = svmModel.predict(Xtrain)
	smallSampleTrainScore = accuracy_score(ypreds, ytrain)

	print "100 train -- TRAINING ACCURACY: " + str(smallSampleTrainScore)




