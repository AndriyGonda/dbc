import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
import numpy as np
import math


def dbc(img,s):
    (width, height) = img.size
    # check width == height
    assert(width == height)
    pixel = img.load()
    M = width
    # grid size must be bigger than 2 and least than M/2
    G = 256
    assert(s >= 2)
    assert(s <= M//2)
    ngrid = math.ceil(M / s)
    h = G*(s / M) # box height
    grid = np.zeros((ngrid,ngrid), dtype='int32')
    
    for i in range(ngrid):
        for j in range(ngrid):
            maxg = 0
            ming = 255
            for k in range(i*s, min((i+1)*s, M)):
                for l in range(j*s, min((j+1)*s, M)):
                    if pixel[k, l] > maxg:
                        maxg = pixel[k, l]

                    if pixel[k, l] < ming:
                        ming = pixel[k, l]
                        
            grid[i,j] = math.ceil(maxg/h) - math.ceil(ming/h) + 1

    Ns = 0
    for i in range(ngrid):
        for j in range(ngrid):
            Ns += grid[i, j]
    return Ns


def sdbc(img,s):
    (width, height) = img.size
    # check width == height
    assert(width == height)
    pixel = img.load()
    M = width
    # grid size must be bigger than 2 and least than M/2
    G = 256
    assert(s >= 2)
    assert(s <= M//2)
    ngrid = math.ceil(M / s)
    h = G*(s / M) # box height
    grid = np.zeros((ngrid,ngrid), dtype='int32')
    
    for i in range(ngrid):
        for j in range(ngrid):
            maxg = 0
            ming = 255
            for k in range(i*s, min((i+1)*s, M)):
                for l in range(j*s, min((j+1)*s, M)):
                    if pixel[k, l] > maxg:
                        maxg = pixel[k, l]

                    if pixel[k, l] < ming:
                        ming = pixel[k, l]
                        
            grid[i,j] = math.ceil((maxg-ming+1)/h)

    Ns = 0
    for i in range(ngrid):
        for j in range(ngrid):
            Ns += grid[i, j]
    return Ns
    


if __name__ == '__main__':
    path = str(input("Enter path to image:"))
    image = Image.open(path) # Brodatz/D1.gif
    image = image.convert('L')
    (imM, _) = image.size
    
    # calculate Nr and r
    Nr = []
    r = []
    print("|\tNr\t|\tr\t|S\t|")
    a = 2
    b = imM//2
    nval = 20
    lnsp = np.linspace(1,math.log(b,a),nval)
    sval  = a**lnsp
	
    for S in sval:#range(2,imM//2,(imM//2-2)//100):
        Ns = dbc(image, int(S))
        Nr.append(Ns)
        R = S/imM
        r.append(S)
        print("|%10d\t|%10f\t|%4d\t|"% (Ns,R,S))
	
	
    # calculate log(Nr) and log(1/r)    
    y = np.log(np.array(Nr))
    x = np.log(1/np.array(r))
    (D, b) = np.polyfit(x, y, deg=1)
    
    # search fit error value
    N = len(x)
    Sum = 0
    for i in range(N):
        Sum += (D*x[i] + b - y[i])**2
        
    errorfit = (1/N)*math.sqrt(Sum/(1+D**2))
    
    # figure size 10x5 inches
    plt.figure(1,figsize=(10,5)).canvas.set_window_title('Fractal Dimension Calculate')
    plt.subplots_adjust(left=0.04,right=0.98)
    plt.subplot(121)
    plt.title(path)
    plt.imshow(image)
    plt.axis('off')

    
    plt.subplot(122)  
    plt.title('Fractal dimension = %f\n Fit Error = %f' % (D,errorfit))
    
    plt.plot(x, y, 'ro',label='Calculated points')
    plt.plot(x, D*x+b, 'k--', label='Linear fit' )
    plt.legend(loc=4)
    plt.xlabel('log(1/r)')
    plt.ylabel('log(Nr)')
    plt.show()

