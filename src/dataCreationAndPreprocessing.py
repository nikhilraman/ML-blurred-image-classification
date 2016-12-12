"""
============================================================
File for converting dataset images to feature vectors 
============================================================

""" 

print(__doc__) 

#### IMPORTS #### 

import numpy as np 
from PIL import Image
import sys 
import os
import scipy 
from scipy import ndimage
#import matplotlib.pyplot as plt 

def getFeatureVector(img):

	g_y, g_x = np.gradient(img) 

	magG = np.hypot(g_x, g_y) 

	# print magG 

	direction = np.arctan2(g_y, g_x) 
	dirDegrees = np.degrees(direction) 

	gradHistProbs, _ = np.histogram(magG, bins=200, range=(0.0, 200.0), density=True) 
	#gradHistProbs, _ = np.histogram(magG, bins=255, range=(0.0, 255.0), density=True)  
	gradHistDegreesProbs, _ = np.histogram(dirDegrees, bins=360, range=(-180.0, 180.0), density=True)   

	print "MAGNITUDE-NP.GRADIENT: Max: " + str(np.max(magG)) + "     Min: " + str(np.min(magG))
	print "DIRECTION: Max: " + str(np.max(dirDegrees)) + "     Min: " + str(np.min(dirDegrees)) 
	
	#newInstanceProbs = np.append(gradHistProbs, gradHistDegreesProbs) 
	newInstanceProbs = gradHistProbs
	return newInstanceProbs 

def processImages(dirName):  

	X = None

	#desiredImages = dirName

	for filename in os.listdir(dirName):  
		full_filename = dirName + "/" + filename;

		imgL = scipy.misc.imread(full_filename, mode='L') 
		img = imgL.astype(float)

		newFeatureVector = np.asmatrix(getFeatureVector(img)) 

		#print newFeatureVector

		if X is None: 
			X = newFeatureVector
		else: 
			X = np.concatenate((X, newFeatureVector), axis=0)

	return X



if __name__ == "__main__": 


	#f_train = open("training-images-small", "a")
	#f_train.close()

	print "Starting!"

	blurTrainingImages = "blur-small-train"
	X_blur = processImages(blurTrainingImages)
	np.savetxt("training-blur-small.dat", X_blur, delimiter=',') 
	#np.savetxt("prints.dat", X_blur, delimiter=',') 

	print "Finished Blur Training"

	noBlurTrainingImages = "no-blur-small-train"
	X_no_blur = processImages(noBlurTrainingImages)
	np.savetxt("training-no-blur-small.dat", X_no_blur, delimiter=',') 
	#np.savetxt("prints.dat", X_no_blur, delimiter=',') 

	print "Finished No Blur Training"

	blurTestImages = "blur-small-test"
	X_blur_test = processImages(blurTestImages)
	np.savetxt("test-blur-small.dat", X_blur_test, delimiter=',') 

	print "Finished Blur Test"

	noBlurTestImages = "no-blur-small-test"
	X_no_blur_test = processImages(noBlurTestImages)
	np.savetxt("test-no-blur-small.dat", X_no_blur_test, delimiter=',')   

	print "Finished No Blur Test"

	print "Done!"




	
	



