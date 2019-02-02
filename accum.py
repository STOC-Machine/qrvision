import cv2
import glob
import numpy as np
import sys
import math
        
def accumulate(image, theta_buckets,rho_buckets):
    
    accum=np.zeros((rho_buckets,theta_buckets))
    
    max_rho=math.sqrt(image.shape[0]*image.shape[0]+image.shape[1]*image.shape[1])
    
    iterator = np.nditer(image, flags=['multi_index'])
    while(not iterator.finished):
        if(iterator[0]!=0):
            print(iterator.multi_index)
            for i in range(0,theta_buckets):
                theta=(2*np.pi*i)/(1.0*theta_buckets)
                rho=(iterator.multi_index[1]*math.cos(theta))+(iterator.multi_index[0]*math.sin(theta))
                j=int((rho+max_rho)/(2*max_rho/(1.0*rho_buckets)))
                accum[j][i]+=1
        iterator.iternext()
    return accum

files=glob.glob(sys.argv[1])
while len(files) > 0:
        file = files.pop(0)
        img = cv2.imread(file)

        if img is None:
            print('Error: could not read image file {}, skipping.'.format(file))
            continue

        cv2.imshow("Hi",img)
        edges=cv2.Canny(img,100,100)
        cv2.imshow("Edges",edges)
        accumulated=accumulate(edges,50,50)
        maximum=np.amax(accumulated)
        
        accumimage=(255.0/maximum)*accumulated
        cv2.imshow("Accum",accumimage.astype(np.uint8))
        # Wait for keypress to continue, close old windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


