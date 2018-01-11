import os
from os import walk, getcwd
from PIL import Image
import cv2 as cv
import numpy as np
import glob
from base64 import b64encode, b64decode
from pascal_voc_io import PascalVocWriter
from pascal_voc_io import XML_EXT
import os.path
import sys


filepath = "NFPAdataset"
outputpath = "ouputImages"
outputlabel = "outputLabels"
for filename in glob.glob(os.path.join(filepath,'*.txt')):
	f = open(filename,"r")
	im = Image.open(filename[:len(filename)-4] + ".jpg")
	img = cv.imread(filename[:len(filename)-4] + ".jpg")
	
	imgFolderPath = filepath
	imgFolderName = os.path.split(imgFolderPath)[-1]
	imgFileName = os.path.basename(filename[:len(filename)-4] + ".jpg")
	imagePath = filename[:len(filename)-4] + ".jpg"
	imageShape = [int(im.size[0]),int(im.size[1]),3]
	
	writer = PascalVocWriter(imgFolderName,imgFileName,imageShape,localImgPath=imagePath)
	
	for line in f:
		cord = line.split()
		x = float(cord[1])
		y = float(cord[2])
		w = float(cord[3])
		h = float(cord[4])
		s0 = int(im.size[0])
		s1 = int(im.size[1])
		xmax = (int)((s0*(2*x+w))/2)
		xmin = (int)((s0*(2*x-w))/2)
		ymax = (int)((s1*(2*y+h))/2)
		ymin = (int)((s1*(2*y-h))/2)
		img = cv.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),4)
		
		bndbox = [xmin,ymin,xmax,ymax]
		writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], 0, 0)
	
	writer.save(targetFile = imgFileName[:len(imgFileName)-4] + ".xml")
	cv.imwrite(outputpath + filename[:len(filename)-4] + ".jpg",img)
	