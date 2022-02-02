'''
Created on 11/02/2015

@author: Andrew Chalmers
'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage
from scipy.ndimage.interpolation import zoom
from sklearn.cluster import k_means
from sklearn.neighbors import NearestNeighbors

def plotImg(img1, ave1, centroids, k, fn = ''):
    fig = plt.figure(1, figsize=(12, 6))
    
    plt.clf()    
    ax = fig.add_subplot(121)
    plt.imshow(img1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    '''
    ax = fig.add_subplot(k+1,2,2)
    plt.imshow(ave1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    '''
    
    for i in range(1, k+1):
        ax = fig.add_subplot(k+1,2,i*2)
        
        AveVal = np.ones((50,100,3))
        AveVal[:,:,0] *= centroids[i-1,0]
        AveVal[:,:,1] *= centroids[i-1,1]
        AveVal[:,:,2] *= centroids[i-1,2]
        
        plt.imshow(AveVal)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
    
    plt.savefig('_'+fn+'.jpg',bbox_inches='tight')     
    plt.show()
    
def getNearest(point, points, k):
    nbrs = NearestNeighbors(k).fit(points)
    distances, indices = nbrs.kneighbors(point.reshape(1, -1))
    return distances, indices

if __name__ == '__main__':
    print('Colour Palette')
    
    # Read in image data
    fnRaw = 'photo3'
    fn = 'images/'+fnRaw+'.jpg'
    
    img = Image.open(fn)
    mat = np.asarray(img) / 255.
    
    width = mat.shape[1]
    height = mat.shape[0]
    numPxls = width*height

    print('Resolution:', width, 'x', height)
    
    red = mat[:,:,0]
    green = mat[:,:,1]
    blue = mat[:,:,2]
    
    rescale = 0.5
    red = zoom(red, rescale)
    green = zoom(green, rescale)
    blue = zoom(blue, rescale)
    mat = np.zeros((int(height*rescale), int(width*rescale), 3))
    
    mat[:,:,0] = red
    mat[:,:,1] = green
    mat[:,:,2] = blue
    
    width = mat.shape[1]
    height = mat.shape[0]
    numPxls = width*height
    
    print('Resized to:', width, 'x', height)
    
    redVec = red.reshape(-1)
    greenVec = green.reshape(-1)
    blueVec = blue.reshape(-1)
    
    imgVec = np.zeros((numPxls,3))
    imgVec[:,0] = redVec
    imgVec[:,1] = greenVec
    imgVec[:,2] = blueVec
    
    # Do analysis    
    # Simple average
    aveImg = np.ones((50,100,3))
    aveImg[:,:,0] *= np.mean(redVec)
    aveImg[:,:,1] *= np.mean(greenVec)
    aveImg[:,:,2] *= np.mean(blueVec)
    
    # Clustering and nearest neighbours
    k = 6
    centroids, labels, intertia = k_means(imgVec, k)
    
    # Optional, helps increase accuracy by taking actual value from sample instead of average
    '''
    for i in range(0, len(centroids)):
        distances, indices = getNearest(centroids[i], imgVec, 1)
        centroids[i] = imgVec[indices[0]][0]  
    '''
    
    # Display results
    plotImg(img, aveImg, centroids, k, fnRaw)
    
    print('Success')
