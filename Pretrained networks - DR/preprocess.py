import cv2, glob, numpy
import glob2
import os
dirname="E:/sabooranew/new/data augmantation/preprocess/trainaug1/normal"
os.chdir(dirname)
def scaleRadius(img,scale):
    x=img[img.shape[0]/2,:,:].sum(1)
    r=(x>x.mean()/10).sum()/2
    s=scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)
scale=300
for f in (glob2.glob("E:/sabooranew/new/data augmantation/preprocess/trainaug1/normal/**/*.jpg")):
        try:
            a=cv2.imread(f)
            a=scaleRadius(a,scale)
            b=numpy.zeros(a.shape)
            cv2.circle(b,(a.shape[1]/2,a.shape[0]/2),int(scale*0.9),(1,1,1),-1,8,0)
            aa=cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)*b+128*(1-b)
            cv2.imwrite(f,aa)       
        except:
            print f
  
