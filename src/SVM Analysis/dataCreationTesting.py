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

	direction = np.arctan2(dy, dx) 
	dirDegrees = np.degrees(direction)  

	##### PRINT STUFF ##### 

	print "MAGNITUDE: Max: " + str(np.max(mag)) + "     Min: " + str(np.min(mag)) 
	print "MAGNITUDE-NORM: Max: " + str(np.max(norm)) + "     Min: " + str(np.min(norm)) 
	print "MAGNITUDE-NP.GRADIENT: Max: " + str(np.max(magG)) + "     Min: " + str(np.min(magG))
	print "DIRECTION: Max: " + str(np.max(dirDegrees)) + "     Min: " + str(np.min(dirDegrees)) 

	#### Histogramming ##### 

	



