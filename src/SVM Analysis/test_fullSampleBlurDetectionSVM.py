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
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__": 

	# Main code  

	##################################
	##### Load the training data ##### 
	##################################

	print "Loading in Training Data"

	## Blur ##
	filename = 'data/training-blur-full-d.dat' 
	data = loadtxt(filename, delimiter=',') 
	Xtrain_blur = data 

	n,d  = Xtrain_blur.shape  
	print n
	ytrain_blur = np.ones(n) 

	## No Blur ##
	filename2 = 'data/training-no-blur-full-d.dat' 
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
	filename3 = 'data/test-blur-full-d.dat' 
	data3 = loadtxt(filename3, delimiter=',') 
	Xtest_blur = data3 

	n3,d  = Xtest_blur.shape 
	print n3
	ytest_blur = np.ones(n3) 

	## No Blur ##
	filename4 = 'data/test-no-blur-full-d.dat' 
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

	# svmModel = svm.SVC(kernel="rbf")  

	# C = 290 		-- Full dataset, just magnitudes, np.gradient
	# gamma = 26	-- 0.70 accuracy

	# C = 300 		-- Full dataset, magnitudes and directions, np.gradient
	# gamma = 3	-- 0.79 accuracy


	# svr =  svm.SVC(kernel='rbf') 
	# param_grid1 = {'C': [ 250, 290, 300, 310, 350, 400, ], 'gamma': [20, 23, 24, 25, 26, 27, 28, 30, 35, 40,], 'kernel': ['rbf']}
	# param_grid1 = {'C': [100, 280, 285, 290, 295, 300, 305, 310, 315, 320, 400], 'gamma': [ 0.3, 0.6, 1, 2,2.5,3,3.5, 4,5,6, 10], 'kernel': ['rbf']}
	# svmModel = GridSearchCV(svr, param_grid = param_grid1, cv = 8) # cv = 3 
	#clf.fit(Xtrain, ytrain)    

	svmModel = svm.SVC(C=310, kernel="rbf", gamma=3)
	svmModel.fit(Xtrain, ytrain)  

	# print "Best score:"
	# print svmModel.best_score_  
	# print "Best params:"
	# print svmModel.best_params_ 

	# svmModel.fit(Xtrain, ytrain)  

	print "Predicting Labels"

	ypreds = svmModel.predict(Xtest) 
	print ypreds

	smallSampleAccuracyScore = accuracy_score(ypreds, ytest)

	print "Full -- TEST ACCURACY: " + str(smallSampleAccuracyScore) 

	ypreds = svmModel.predict(Xtrain)
	smallSampleTrainScore = accuracy_score(ypreds, ytrain)

	print "Full -- TRAINING ACCURACY: " + str(smallSampleTrainScore)




