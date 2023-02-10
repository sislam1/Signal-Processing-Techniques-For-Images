"""
Author: Luca Giancardo
Date: 2017-09-28
Version: 1.0

Author: Samia Islam
Date: 2022-09-20
Version: 1.1
"""

# import libraries
import matplotlib.pyplot as plt

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform as sktr
import skimage.filters as skfl
import skimage.feature as skft
import skimage.color as skcol
import skimage.exposure as skexp
import skimage.morphology as skmr 
import scipy.ndimage as ndimage

FS = 1000 # number of samples in the x domain DOMAIN = [-2, 2] # extent of the x domain
DOMAIN = [-2, 2] # extent of the x domain

class Freq:
    def __init__(self):
        print('Class instantiated\n')

    def sin2(self, x, freq=1, phase=0, amplitude=1):
        """
        Compute sine wave with given frequency, phase and amplitude

        :param x: array of the domain expressed in sine cycles, i.e. x=1 represent the sine of 360 degrees (or 2pi),
        :param freq: sine frequency
        :param phase: sine phase
        :param amplitude: sine amplitude
        :return: array of values
        """
        y = amplitude * np.sin(x * 2 * np.pi * freq - (phase * 2 * np.pi))
        return y
    
    def sumy(self, y):
        # create 2D numpy array from y
        y_matrix = np.array(y)
        yAll = y_matrix.sum(axis=0)
        return yAll

    def fft1D(self, xIn, yIn):
        """
        Given a signal y(x) compute the FFT of a 1D signal and return the magnitude and phase at each frequency
        :param xIn: x of signal
        :param yIn: y of signal
        :return: (freqArray, spMag, spPhase)
        """
        # initialization
        domain = (min(xIn),max(xIn))
        fs = len(xIn)/2

        # Compute FFT
        sp = np.fft.fft(yIn) / len(xIn)
        # Compute frequency array
        freqArray = np.fft.fftfreq(len(xIn), (domain[1] - domain[0]) / 2 / fs)
        # Shift FFT to interpretable format
        sp = np.fft.fftshift(sp)
        freqArray = np.fft.fftshift(freqArray)

        # compute magnitude and angle of complex number
        spMag = np.absolute(sp)
        spPhase = np.angle(sp)

        return freqArray, spMag, spPhase

    def plotFFT2d(self, imgIn, sizeFig=(12, 6)):
        """
        Compute the FFT of a 2D signal and diplay magnitude component
        :param imgIn: 2d signal
        :param sizeFig: optional tuple with plot size
        :return: none
        """
        # resolution
        M, N = imgIn.shape

        # compute FFT
        F = np.fft.fftn(imgIn)
        # compute magnitude and shift FFT to interpretable format
        F_magnitude = np.abs(F)
        F_magnitude = np.fft.fftshift(F_magnitude)

        # plot
        plt.subplots(figsize=sizeFig)
        plt.imshow(np.log(1 + F_magnitude), cmap='viridis', extent=(-N // 2, N // 2, -M // 2, M // 2))
        plt.colorbar()
        plt.title('Spectrum magnitude');
        plt.show()

    # Compute sin waves and plot
    def siny(self, param_array):
        #a3freq = a3.Freq()
        # set up x axis
        x = np.linspace(DOMAIN[0], DOMAIN[1], 2*FS)
        # set up array to store freq, phase, amplitude, respetively, for each wave that will be plotted
        #param_ary = [[1, 0, 1], [1, 0.25, 0.5], [3, 0, 0.25]]
        # create an empty array y. Each element of y will store all the y output values based on
        # length of x. Length of y is the number of sets of parameters we are computing, i.e. length
        # of param_ary
        y =  [[] * len(x)] * len(param_array)
        return x, y

    def ploty(self, x, y, param_array):
        # loop through param_ary
        for i in range(len(param_array)):
            y[i] = self.sin2(x, freq=param_array[i][0], phase=param_array[i][1], amplitude=param_array[i][2])
            plt.plot(x, y[i], label='freq={0}, phase={1}, amplitude={2}'.format(param_array[i][0], param_array[i][1], param_array[i][2]))
        plt.title('Sin Wave Signals')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend(fontsize=6)
        plt.show()
    
    # sum all sin wave signals
    def sumyAll(self, x, y):
        yAll = self.sumy(y)
        return yAll

    def plotFunc(self, x, y):
        plt.plot(x, y)
        plt.title('Sum of Sin Wave Signals')
        plt.xlabel('x')
        plt.ylabel('yAll')
        plt.grid(True)
        plt.show()
    
    # plot FFT
    def plotFFT(self, freqArray, spMag):
        # plot FFT magnitude vs frequency for frequencies between -5 and 5 Hz
        bool_freq = (freqArray>-5) & (freqArray<5)
        plt.xlabel('frequency')
        plt.ylabel('magnitude')
        plt.title('FFT of yAll')
        plt.stem(freqArray[bool_freq], abs(spMag[bool_freq]))
        plt.show()

# this class performs all functions regarding processing 
# the image
class Image_process:
    def __init__(self, image_in, multIm=False, templateIm=None):
        """
        Constructor
        """
        self.image_in = image_in
        self.multIm = multIm
        self.templateIm = templateIm
    
    # load image file and resize to 20%. Return resized image
    def load_image(self):
        # check if 'data' is in current directory or in 
        # ../ directory
        # check if image is single image or multiple images
        if self.image_in:
            im_orig = skio.imread(self.image_in)
            if self.multIm:
                im_template = skio.imread(self.templateIm)
                return im_orig, im_template
            else:
                return im_orig
        else:
            im_orig = skio.imread('../'+self.image_in)
            if self.multIm:
                im_template = skio.imread('../'+self.templateIm)
                return im_orig, im_template
            else:
                return im_orig

    # resize image to 20%
    def resz_image(self, img):    
        im_resz = sktr.resize(img, (img.shape[0] // 5, img.shape[1] // 5), 
                       anti_aliasing=True)
        #im_resz = sktr.rescale(img, 0.2, multichannel=True)
        return im_resz

    # convert rgb to grayscale and return converted image
    def convrgb2gray(self, img):
        imG = skcol.rgb2gray(img)
        return imG

    # extract window around optic nerve which is 20% of image height and centered
    # around coordinates (313, 357)
    def extractWindow(self, img):
        # 20% of height
        borderPx = int(img.shape[0] * 0.20)
        # coordinate of ON
        coord = (313, 357)
        # slice image
        imONest = img[coord[0]-borderPx:coord[0]+borderPx, coord[1]-borderPx:coord[1]+borderPx]
        return imONest

    # find horizontal and vertical edges of image using Sobel transform and return
    def horVerImg(self, img):
        return skfl.sobel_h(img), skfl.sobel_v(img)
    
    # find magnitude of edges. Input are horizontal and vertical gradients obtained
    # through Sobel transform
    def magEdges(self, img_hor, img_ver):
        L_xy = (img_hor**2+img_ver**2)**0.5
        return L_xy
    
    def subplots(self, img_hor, img_ver, L_xy):
        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)
        plt.title('Horizontal Gradient')
        plt.imshow(img_hor, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title('Vertical Gradient')
        plt.imshow(img_ver, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title('Edge strength \n(magnitude of gradients')
        plt.imshow(L_xy, cmap='gray')
        plt.show()

    # perform convolution of image by manually defining Sobel filter
    def convSob(self, img):
        sobelHfilt = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/4.0
        convRes = ndimage.convolve(img, sobelHfilt)
        return convRes

    # generate Gaussian and salt and pepper noise
    def genNoise(self, img, gauss=1):
        if gauss: # generate gaussian noise
            noise = np.random.normal(np.zeros(img.shape), scale=0.02)
        else:
            # Salt end pepper noise
            noiseSalt = np.random.normal(np.zeros(img.shape), scale=1) > 2 
            noisePepper = (np.random.normal(np.zeros(img.shape), scale=1) > 2) * -1 
            noise = (((noiseSalt + noisePepper) / 2.) + 0.5)
        return noise

    def imgWithNoise(self, img, gauss=1):
        imgNoise = img.copy() # first copy image
        # Generate noise and add noise to image
        if gauss: # gaussian
            noise = self.genNoise(img, gauss=1)
            imgNoise = imgNoise + noise
        # this will be for salt and pepper. But if more types of 
        # noises are added later on, change this to conditional
        # elif statement
        else: # salt and pepper
            noise = self.genNoise(img, gauss=0)
            # set mask for noise coordinates
            noiseCoord = (noise > 0.9) | (noise < 0.1) 
            imgNoise[noiseCoord] = noise[noiseCoord]
        return imgNoise, noise
    
    # display gaussian and salt and pepper noise, along with images
    # corrupted by the noise
    def dispNoise(self, noiseGauss, noiseSnP, imgNoiseGauss, imgNoiseSnP):
        figure, axis = plt.subplots(2, 2, figsize=(20,20))
        im0 = axis[0,0].imshow(noiseGauss, cmap='gray')
        figure.colorbar(im0, ax=axis[0, 0])
        axis[0,0].set_title('Gaussian noise')
        im1 = axis[0,1].imshow(noiseSnP, cmap='gray')
        figure.colorbar(im1, ax=axis[1, 0])
        axis[0,1].set_title('Salt and Pepper noise')
        im2 = axis[1,0].imshow(imgNoiseGauss, cmap='gray', vmin=0, vmax=1)
        figure.colorbar(im2, ax=axis[0, 1])
        axis[1,0].set_title('image corrupted with gaussian noise')
        im3 = axis[1,1].imshow(imgNoiseSnP, cmap='gray', vmin=0, vmax=1)
        figure.colorbar(im3, ax=axis[1, 1])
        axis[1,1].set_title('image corrupted with salt and pepper noise')
        plt.show()
    
    # pass image through filter
    def filterImage(self, img, gauss=1):
        if gauss: # gaussian filter
            return skfl.gaussian(img)
        # this will be for median filter. But if more types of 
        # filters are added later on, change this to conditional
        # elif statement
        else:
            resIm = skfl.median(sk.img_as_uint(img), skmr.square(3))
            # median filter needs an image represented as integers. So
            # turn the result back to float
            #print(sk.img_as_float(resIm))
            return sk.img_as_float(resIm)

    # apply gaussian and median filtering to images corrupted with gaussian
    # and salt and pepper noise
    def gaussMedianFilter(self, imgNoiseGauss, imgNoiseSnP):
        imResGaussGauss = self.filterImage(imgNoiseGauss, gauss=1)
        imResGaussMedian = self.filterImage(imgNoiseGauss, gauss=0)
        imResSnPGauss = self.filterImage(imgNoiseSnP, gauss=1)
        imResSnPMedian = self.filterImage(imgNoiseSnP, gauss=0)
        figure, axis = plt.subplots(2, 3, figsize=(20,20))
        img0 = axis[0,0].imshow(imgNoiseGauss, cmap='gray')
        figure.colorbar(img0, ax=axis[0, 0])
        axis[0,0].set_title('Image corrupted by gaussian noise')
        img1 = axis[0,1].imshow(imResGaussGauss, cmap='gray')
        figure.colorbar(img1, ax=axis[0, 1])
        axis[0,1].set_title('Gaussian noise removed by gaussian filter')
        img2 = axis[0,2].imshow(imResGaussMedian, cmap='gray')
        figure.colorbar(img2, ax=axis[0, 2])
        axis[0,2].set_title('Gaussian noise removed by median filter')
        img3 = axis[1,0].imshow(imgNoiseSnP, cmap='gray')
        figure.colorbar(img3, ax=axis[1, 0])
        axis[1,0].set_title('Image corrupted by salt and pepper noise')
        img4 = axis[1,1].imshow(imResSnPGauss, cmap='gray')
        figure.colorbar(img4, ax=axis[1, 1])
        axis[1,1].set_title('Salt and pepper noise \nremoved by gaussian filter')
        img5 = axis[1,2].imshow(imResSnPMedian, cmap='gray')
        figure.colorbar(img5, ax=axis[1, 2])
        axis[1,2].set_title('Salt and pepper noise \nremoved by median filter')
        plt.show()
    
    def matchTemplate(self, img, imgt):
        return skft.match_template(img, imgt)

if __name__ == "__main__":
    pass

