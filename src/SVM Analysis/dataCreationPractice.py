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

#### STEPS #### 

# 1. load image 
# 2. convert to grayscale 
# 3. take vertical "derivative" (direction of maximum intensity increase) for each pixel
# 			--> we will model this as maximum difference along y-axis = g_y
# 4. take horizontal "derivative" (direction of maximum intensity increase) for eaxh pixel
# 			--> we will model this as maximum difference along x-axis = g_x
# 5. assign each pixel gradient magnitude = sqrt(g_y^2 + g_x^2) || gradient direction = arctan(g_y/g_x)
# 6. Round each value to nearest integer and create histogram of values for magnitude -> this becomes vector of length n1 
# 		 	--> where n1 is 300 (or 256 or 200 or the actual max)
# 7. Round each value to nearest integer and create histogram of values for direction -> this becomes vector of length n2 
#			--> where n2 is 360 (ranging from -180 to 180 for degrees) 
# 8. Concatenate the two vectors and we have our desired feature vector 


if __name__ == "__main__": 

	# Main code  

	### Print the command line arguments for verifcation ###
	print 'Number of arguments:', len(sys.argv), 'arguments.'
	print 'Argument List:', str(sys.argv), '\n' 

	infile = sys.argv[1] 

	### Open the input file and convert to matrix of pixels ###

	# img = None
	# try:
	# 	img = Image.open(infile)
	# except IOError:
	# 	print "Invalid file!", infile
	# 	print "Please Enter a command of the form: "
	# 	print "\n'python imageSegmentation.py K inputImageFilename outputImageFilename'\n"
	# 	sys.exit() 


	# X = np.asarray(list(img.getdata()))
	# width, height = img.size    

	#img = scipy.misc.imread(infile)
	imgL = scipy.misc.imread(infile, mode='L')  
	#imgL = scipy.misc.imread(infile) 

	print "Shape:"
	print imgL.shape

	#print img 
	print "NEXT -----------------" 
	print imgL

	#print X


	#img = imgL.astype('int32') 
	img = imgL.astype(float)
	dx = ndimage.sobel(img, 1) 
	#print dx
	dy = ndimage.sobel(img, 0) 
	#print dy
	mag = np.hypot(dx, dy) 

	print "MAGNITUDE: Max: " + str(np.max(mag)) + "     Min: " + str(np.min(mag))
	#maxN = np.max(mag)
	#print maxN
	norm = mag * (255.0 / np.max(mag))
	#mag *= 255.0 / np.max(mag) 

	print "MAGNITUDE-NORM: Max: " + str(np.max(norm)) + "     Min: " + str(np.min(norm))

	orient = np.arctan2(dy, dx) 
	orient = np.degrees(orient)  
	print "DIRECTION: Max: " + str(np.max(orient)) + "     Min: " + str(np.min(orient))
	
	print "Shape mag " 
	print mag.shape
	print norm.shape 


	gradient_y, gradient_x = np.gradient(img)

	magG = np.hypot(gradient_x, gradient_y) 
	print "MAGNITUDE-NUMPY: Max: " + str(np.max(magG)) + "     Min: " + str(np.min(magG)) 

	direction = np.arctan2(gradient_y, gradient_x) 
	dirDegrees = np.degrees(direction)  
	print "DIRECTION: Max: " + str(np.max(dirDegrees)) + "     Min: " + str(np.min(dirDegrees))

	# fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
	# axes[0].imshow(dx, interpolation='none', cmap='gray')
	# axes[0].set(title="Scipy's Sobel")

	# axes[1].imshow(gradient_x, interpolation='none', cmap='gray')
	# axes[1].set(title="Numpy's Gradient")

	# plt.show() 

	# sobel = ndimage.generic_gradient_magnitude(img, ndimage.sobel) 

	# print "MAGNITUDE-SOBEL: Max: " + str(np.max(sobel)) + "     Min: " + str(np.min(sobel))

	# fig, ax = plt.subplots()
	# ax.imshow(magNumpy, interpolation='none', cmap='gray')
	# plt.show() 

	#### Histogramming ##### 

	# sobelHist = np.histogram(norm, bins=255, range=(0.0, 255.0)) 
	# print sobelHist
	# gradHist = np.histogram(magG, bins=255, range=(0.0, 255.0)) 
	# print gradHist

	# scipy.misc.imsave('blur-city-dx.jpg', dx)
	# scipy.misc.imsave('blur-city-dy.jpg', dy)
	# scipy.misc.imsave('blur-city-magnitude.jpg', mag)







