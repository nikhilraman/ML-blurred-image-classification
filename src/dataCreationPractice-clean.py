"""
============================================================
File for practicing converting an image to a feature vector 
============================================================

""" 

print(__doc__) 

#### IMPORTS #### 

import numpy as np 
from PIL import Image
import sys 
import scipy 
from scipy import ndimage
import matplotlib.pyplot as plt 


if __name__ == "__main__": 

	# Main code  

	### Print the command line arguments for verifcation ###
	print 'Number of arguments:', len(sys.argv), 'arguments.'
	print 'Argument List:', str(sys.argv), '\n' 

	infile = sys.argv[1] 

	imgL = scipy.misc.imread(infile, mode='L')   

	print "Shape:"
	print imgL.shape

	#print img 
	print "NEXT -----------------" 
	print imgL 
	print "NEXT -----------------"  

	###### SOBEL SCIPY ########

	#img = imgL.astype('int32') 
	img = imgL.astype(float)
	dx = ndimage.sobel(img, 1) 
	dy = ndimage.sobel(img, 0) 
	mag = np.hypot(dx, dy)  

	norm = mag * (255.0 / np.max(mag)) 

	orient = np.arctan2(dy, dx) 
	orient = np.degrees(orient) 

	###### GRADIENT NUMPY ########

	g_y, g_x = np.gradient(img) 

	magG = np.hypot(g_x, g_y)  

	direction = np.arctan2(g_y, g_x) 
	dirDegrees = np.degrees(direction)  

	##### PRINT STUFF ##### 

	print "MAGNITUDE: Max: " + str(np.max(mag)) + "     Min: " + str(np.min(mag)) 
	print "MAGNITUDE-NORM: Max: " + str(np.max(norm)) + "     Min: " + str(np.min(norm)) 
	print "MAGNITUDE-NP.GRADIENT: Max: " + str(np.max(magG)) + "     Min: " + str(np.min(magG))
	print "DIRECTION: Max: " + str(np.max(dirDegrees)) + "     Min: " + str(np.min(dirDegrees)) 

	#### Histogramming ##### 

	print "HISTOGRAMZZZ"

	
	sobelHist, _ = np.histogram(norm, bins=255, range=(0.0, 255.0))  
	print "HISTOGRAMZZZ -- SOBEL MAGS"
	print sobelHist 
	
	# plt.hist(norm, bins=255, range=(0.0, 255.0)) 
	# plt.title("HISTOGRAMZZZ -- SOBEL MAGS")
	# plt.show()

	gradHist, _ = np.histogram(magG, bins=255, range=(0.0, 255.0))  
	gradHistProbs, _ = np.histogram(magG, bins=255, range=(0.0, 255.0), density=True)  
	

	print "HISTOGRAMZZZ -- GRADIENT MAGS"
	print gradHist  
	# plt.hist(magG, bins=255, range=(0.0, 255.0)) 
	# plt.title("HISTOGRAMZZZ -- GRADIENT MAGS")
	# plt.show

	sobelHistDegrees, _ = np.histogram(orient, bins=360, range=(-180.0, 180.0))
	print "HISTOGRAMZZZ -- SOBEL DIRS" 
	print sobelHistDegrees 
	
	gradHistDegrees, _ = np.histogram(dirDegrees, bins=360, range=(-180.0, 180.0))  
	gradHistDegreesProbs, _ = np.histogram(dirDegrees, bins=360, range=(-180.0, 180.0), density=True)  
	

	print "HISTOGRAMZZZ -- GRADIENT DIRS"
	print gradHistDegrees 


	newInstance = np.append(gradHist, gradHistDegrees) 
	newInstanceProbs = np.append(gradHistProbs, gradHistDegreesProbs)

	print newInstanceProbs

	print newInstance
	print newInstance.size  

	matX = np.asmatrix(newInstance)
	test = np.concatenate((matX, matX), axis=0)

	f = open("test.dat", "a") 
	np.savetxt(f, test, fmt='%d', delimiter=',')
	#np.savetxt(f, newInstance, fmt='%d', delimiter=',', newline=',')
	f.close()
	 




