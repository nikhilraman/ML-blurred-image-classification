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


	dx = ndimage.sobel(img, 1) 
	#print dx
	dy = ndimage.sobel(img, 0) 
	#print dy
	mag = np.hypot(dx, dy) 

	norm = mag * (255.0 / np.max(mag))

	orient = np.arctan2(dy, dx) 
	dirDegrees = np.degrees(orient) 

	gradHistProbs, _ = np.histogram(norm, bins=255, range=(0.0, 255.0), density=True) 
	gradHistDegreesProbs, _ = np.histogram(dirDegrees, bins=360, range=(-180.0, 180.0), density=True) 

	# g_y, g_x = np.gradient(img) 

	# magG = np.hypot(g_x, g_y) 

	# # print magG 

	# direction = np.arctan2(g_y, g_x) 
	# dirDegrees = np.degrees(direction) 

	# gradHistProbs, _ = np.histogram(magG, bins=200, range=(0.0, 200.0), density=True) 
	# #gradHistProbs, _ = np.histogram(magG, bins=255, range=(0.0, 255.0), density=True)  
	# gradHistDegreesProbs, _ = np.histogram(dirDegrees, bins=360, range=(-180.0, 180.0), density=True)   

	# print "MAGNITUDE-NP.GRADIENT: Max: " + str(np.max(magG)) + "     Min: " + str(np.min(magG))
	# print "DIRECTION: Max: " + str(np.max(dirDegrees)) + "     Min: " + str(np.min(dirDegrees)) 
	
	newInstanceProbs = np.append(gradHistProbs, gradHistDegreesProbs) 
	# newInstanceProbs = gradHistProbs
	return newInstanceProbs 

def processImages(dirName):  

	X = None

	#desiredImages = dirName

	count = 0

	for filename in os.listdir(dirName):   
		if (filename.endswith(".jpg") is False) and (filename.endswith(".JPG") is False):
			continue; 

		count = count + 1

		full_filename = dirName + "/" + filename;

		imgL = scipy.misc.imread(full_filename, mode='L') 
		img = imgL.astype(float)

		newFeatureVector = np.asmatrix(getFeatureVector(img)) 

		#print newFeatureVector

		if X is None: 
			X = newFeatureVector
		else: 
			X = np.concatenate((X, newFeatureVector), axis=0)

	print "Num processed: " + str(count)
	return X



if __name__ == "__main__": 


	#f_train = open("training-images-small", "a")
	#f_train.close()

	print "Starting!"

	blurTrainingImages = "data/blur-train-full"
	X_blur = processImages(blurTrainingImages)
	np.savetxt("data/training-blur-full-s.dat", X_blur, delimiter=',') 

	# print "Finished Blur Training"

	noBlurTrainingImages = "data/no-blur-train-full"
	X_no_blur = processImages(noBlurTrainingImages)
	np.savetxt("data/training-no-blur-full-s.dat", X_no_blur, delimiter=',') 

	print "Finished No Blur Training"

	blurTestImages = "data/blur-test-full"
	X_blur_test = processImages(blurTestImages)
	np.savetxt("data/test-blur-full-s.dat", X_blur_test, delimiter=',') 

	print "Finished Blur Test"

	noBlurTestImages = "data/no-blur-test-full"
	X_no_blur_test = processImages(noBlurTestImages)
	np.savetxt("data/test-no-blur-full-s.dat", X_no_blur_test, delimiter=',')   

	print "Finished No Blur Test"

	print "Done!"



