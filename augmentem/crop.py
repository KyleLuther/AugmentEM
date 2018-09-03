import numpy as np

def crop_augment(img, labels, max_z, max_r):
    cz = np.random.randint(-max_z,max_z+1)
    cy = np.random.randint(-max_r,max_r+1)
    cx = np.random.randint(-max_r,max_r+1)

    img = crop(img, cz, cy, cx)
    labels = [crop(l, cz, cy, cx) for l in labels]

    return img, labels
    
def crop(arr, z, y, x):
    Z,Y,X,_ = arr.shape
    c = np.zeros_like(arr)
    c[max(0,z):min(Z,Z+z),max(0,y):min(Y,Y+y),max(0,x):min(X,X+x)] = arr[max(0,-z):min(Z,Z-z),max(0,-y):min(Y,Y-y),max(0,-x):min(X,X-x)]

    return c 
