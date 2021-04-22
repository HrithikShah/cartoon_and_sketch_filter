# CODE

import cv2
import numpy as np

def cartoon(img,ds_factor=4,sketch=False):
    
    # convert image to grayscale
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #APPLY MEDIAN FILTER TO GRAYSCALE
    img_gray=cv2.medianBlur(img_gray,7)
    
    #detect edges in the image and threshold it
    edges=cv2.Laplacian(img_gray,cv2.CV_8U,ksize=5)
    ret,mask=cv2.threshold(edges,100,255,cv2.THRESH_BINARY_INV)
    
    # MASK IS THE SKETCH OF IMAGE
    
    if sketch:
        return cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    
    #RESIZE THE IMAGE TO SAMLLER SIZE FOR FASTER COMPUTATION
    
    img_small=cv2.resize(img,None,fx=1/ds_factor,fy=1/ds_factor,interpolation=cv2.INTER_AREA)
    num_rep=10
    sigmacolor=5
    sigmaspace=7
    size=5
    
    #apply bilateral filter the image multiple times
    
    for i in range(num_rep):
        
        img_small=cv2.bilateralFilter(img_small,size,sigmacolor,sigmaspace)
    
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR)
    dst=np.zeros(img_gray.shape)
    # add the thick boundary
    
    dst=cv2.bitwise_and(img_output,img_output,mask=mask)
    return dst


cap=cv2.VideoCapture(0)
    
cur_char=-1
pre_char=-1
    
while True:
        
    ret,frame=cap.read()
        
    frame=cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
        
    c=cv2.waitKey(1)
    if c==27:
        break
            
    if c>-1 and c!=pre_char:
        cur_char=c
    pre_char=c
        
    if cur_char==ord('s'):
        cv2.imshow("cartoonize",cartoon(frame,sketch=True))
    elif cur_char==ord('c'):
        cv2.imshow("cartoonize",cartoon(frame,sketch=False))
    else:
        cv2.imshow("cartoonize",cartoon(frame))
                       
    
                       
cap.release()
cv2.destroyAllWindows()
                

    
