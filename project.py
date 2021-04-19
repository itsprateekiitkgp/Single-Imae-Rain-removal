import cv2
import numpy as np
import k_svd1 as k_svd
import matplotlib.pyplot as plt
from math import log10, sqrt
from scipy.fftpack import dct as dct2
from scipy.fftpack import idct as idct2

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
def im2col(image,blockSize,step):
    M,N=image.shape
  
    rowNumber=int((M-blockSize)/step)+1
    colNumber=int((N-blockSize)/step)+1
    rows=[i*step for i in range(rowNumber)]
    cols=[i*step for i in range(colNumber)]
    if (rowNumber-1)*step+blockSize<M:
        rows.append(M-blockSize)
    if (colNumber-1)*step+blockSize<N:
        cols.append(N-blockSize)
    repmat=np.array([[image[i:i+blockSize,j:j+blockSize] for j in cols] for i in rows])
   
    repmat=np.reshape(repmat,[len(rows),len(cols),blockSize*blockSize])
    repmat=np.reshape(repmat,[len(rows)*len(cols),blockSize*blockSize])

    return repmat,rows,cols

def main():

    sigma=25
    blockSize=8
    step=1
    maxBlockToTrain=65000
    maxBlockToConsider=260000

    image=cv2.imread('test_image2.jpg',1)
    im_orig=image
    im_orig=cv2.resize(im_orig,(512,512));
    image=im_orig
    image=cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    Y,U,V=cv2.split(image)
    image=Y
    im=Y
    Orig = im
    # Transform
    Orig_T = dct2(Orig);
    # Split between high- and low-frequency in the spectrum (*)
    cutoff = round(0.5 * 256);
    High_T = np.fliplr(np.tril(np.fliplr(Orig_T), cutoff));
    Low_T = Orig_T - High_T;
    # Transform back
    High = idct2(High_T);
    Low = idct2(Low_T);
    
    #cv2.imshow('shuhs',High)
    #cv2.waitKey(0)
    
    #image2=cv2.bilateralFilter(image,5,256,256)
    #cv2.imshow('view',im-image)

    #cv2.waitKey(0);
    #exit()
    #image3=image-image2
    #lpf=image2
    #image=image3

    #cv2.imshow("image",)
    #cv2.waitKey(0)
    
    print(np.shape(image));
    with_rain=image.astype('float');
    #sigma*np.random.randn(*image.shape)

    dataMatrix,_,_=im2col(with_rain,blockSize,step)
    np.random.shuffle(dataMatrix)
    dataMatrix=np.transpose(dataMatrix,[1,0])
   
    if dataMatrix.shape[1]>maxBlockToTrain:
        dataMatrix=dataMatrix[:,:maxBlockToTrain]  # shape [n,N]
    
    #subtract the DC value from the original signal
    mean=np.sum(dataMatrix,0)/dataMatrix.shape[0]
    dataMatrix=dataMatrix-np.tile(mean,[dataMatrix.shape[0],1])
    #construct the k-svd object to do the sparse coding
    dict_list=[]
    for i in range(0,3):
        K=256
        ksvd=k_svd.ksvd(words=K,iteration=3,errGoal=sigma*1.15) # words is the value of k

        #dictionary,dict_list=ksvd.constructDictionary(dataMatrix)
        dictionary,coef=ksvd.constructDictionary(dataMatrix)
        dataMatrix=dataMatrix-np.matmul(dictionary,coef)
        dict_list.append(dictionary)
    print(np.shape(dict_list));
    #print(dictionary);
    print("finish dictionary training")
    x=1
    l=np.shape(dict_list[0])[1]

    SSIM=[]
    print(l)
    dict2=dict_list[0]

    for i in range(0,l):
        new=[]
        for j in range(0,l):
            d1=dict2[:,i:i+1].flatten()
            d2=dict2[:,j:j+1].flatten()

            covar=np.cov(d1,d2)[0][1]
            std1=np.std(d1)
            std2=np.std(d2)

            
            C=1323
            SI=abs(2*covar+C)/abs(std1*std1+std2*std2+C)
            new.append(SI)
        SSIM.append(new)
    anss=[]
    max1=0;
    
    plt.imshow(SSIM, cmap='hot', interpolation='nearest')
    plt.show()

    for x in range(0,2**5):
        s=[]
        for x1 in range(0,K):
            
            if(((1<<x1)&x)!=0):
                s.append(1)
            else:
                s.append(0)
        
        s=np.array(s)
        s=s.reshape((K,1))
        st=s.transpose()
        

        mat1=np.matmul(SSIM-np.identity(K),s)
        mat1=np.matmul(st,mat1)
        val=np.matmul(st,s)
        if(val==0):
            continue;
        ans=(mat1[0][0]*(val**0.2)[0][0])/(val**2)[0][0]
        
        if(ans>max1):
            anss=s
            max1=ans

    for dictionary in dict_list:
        step=1
        #removing rain in the image using the resulted dictionary
        while ((image.shape[0]-blockSize)/step+1)*((image.shape[1]-blockSize)/step+1)>maxBlockToConsider:
            step+=1

        dataMatrix,rowIndex,colIndex=im2col(with_rain,blockSize,step)
        dataMatrix=np.transpose(dataMatrix,[1,0])
        n,N=dataMatrix.shape
        processstep=10000
        maxStep=N//processstep
        if N%processstep:
           maxStep+=1
        for i in range(maxStep):
            maxColumn=np.minimum((i+1)*processstep,N)
            mean=np.sum(dataMatrix[:,i*processstep:maxColumn],0)/n
            dataMatrix[:,i*processstep:maxColumn]-=np.tile(mean,[n,1])
            coef=ksvd.OMP(dictionary,dataMatrix[:,i*processstep:maxColumn],sigma*1.15,showFlag=False)
            dataMatrix[:,i*processstep:maxColumn]=np.dot(dictionary,coef)+np.tile(mean,[n,1])


        imageOut=np.zeros(image.shape)
        weight=np.zeros(image.shape)
        for i,r in enumerate(rowIndex):
            for j,c in enumerate(colIndex):
                block=np.reshape(dataMatrix[:,i*len(rowIndex)+j],[blockSize,blockSize])
                imageOut[r:r+blockSize,c:c+blockSize]+=block
                weight[r:r+blockSize,c:c+blockSize]+=1
        rain_removed=(imageOut/weight).astype(np.uint8)

        cv2.imshow("origin",im_orig)
        cv2.imshow('with_rain',with_rain.astype(np.uint8))
        rain_removed=cv2.merge([rain_removed,U,V]);
        rain_removed=cv2.cvtColor(rain_removed, cv2.COLOR_YUV2BGR)
        cv2.imshow('rain_removed',rain_removed)
        cv2.imwrite('rain_removed_1.jpg',rain_removed)
        cv2.imwrite('original_1.jpg',im_orig)
        cv2.waitKey(10000)
        print("PSNR")
        print(PSNR(im_orig,rain_removed))
        break;

        


if __name__=="__main__":
    main()