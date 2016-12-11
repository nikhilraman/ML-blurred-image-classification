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

	img = None
	try:
		img = Image.open(infile)
	except IOError:
		print "Invalid file!", infile
		print "Please Enter a command of the form: "
		print "\n'python imageSegmentation.py K inputImageFilename outputImageFilename'\n"
		sys.exit() 


	X = np.asarray(list(img.getdata()))
	width, height = img.size   

	print X





