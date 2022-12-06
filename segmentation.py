import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageChops

def is_greyscale(image):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    """
    im = Image.open(image)
    if im.mode not in ("L", "RGB"):
        raise ValueError("Unsuported image mode")

    if im.mode == "RGB":
        rgb = im.split()
        if ImageChops.difference(rgb[0],rgb[1]).getextrema()[1]!=0: 
            return im,False #"Colored Image uploaded"
        if ImageChops.difference(rgb[0],rgb[2]).getextrema()[1]!=0: 
            return im,False #"Colored Image uploaded"
    return im,True #"Grey scale Image uploaded"

def load_data(dir_name):    
    '''
    Load images from the "tempDir" directory If the Image is in RGB then we will convert it to gray scale image
    '''
    for filename in os.listdir(dir_name):
        if os.path.isfile(dir_name + '/' + filename):
            im,check = is_greyscale(dir_name + '/' + filename)
            return im,check

def threshold_otsu(gray_img, nbins=.1):    
    file_name = gray_img.split("/")[-1]
    name,ext = file_name.split(".")
    img = Image.open(gray_img)
    gray_img = np.array(img)

    all_pixels = gray_img.flatten()
    p_all = len(all_pixels)
    least_variance = -1
    least_variance_threshold = -1
    
    # create an array of all possible threshold values which we want to loop through
    color_thresholds = np.arange(np.min(gray_img)+nbins, np.max(gray_img)-nbins, nbins)
    
    # loop through the thresholds to find the one with the least class variance
    for color_threshold in color_thresholds:
        # background
        bg_pixels = all_pixels[all_pixels < color_threshold]
        p_bg = len(bg_pixels)
        w_bg = p_bg / p_all
        variance_bg = np.var(bg_pixels)
        
        # foreground
        fg_pixels = all_pixels[all_pixels >= color_threshold]
        p_fg = len(fg_pixels)
        w_fg = p_fg / p_all
        variance_fg = np.var(fg_pixels)
        
        variance = w_bg * variance_bg + w_fg * variance_fg
        print("trace:", variance, color_threshold)
        
        if least_variance == -1 or variance < least_variance:
            least_variance = variance
            least_variance_threshold = color_threshold

    path = os.getcwd() + "/tempDir/ostu/"

    grey_img = plt.imshow(gray_img,cmap='gray') 
    plt.axis('off')
    plt.title("Gray scale Image")
    plt.savefig(path + name + '_gray' + "." + ext)

    least_variance_threshold = round(least_variance_threshold)
    binary_np_gray_img = 1.0 * (gray_img > least_variance_threshold)
    ixplot = plt.imshow(binary_np_gray_img,cmap='gray') 
    plt.axis('off')
    plt.title(" Threshold = " + str(least_variance_threshold))
    plt.savefig(path + name + '_T' + str(least_variance_threshold) + "." + ext)

    return least_variance_threshold

def threshold_binary(gray_img, threshold):
    file_name = gray_img.split("/")[-1]
    name,ext = file_name.split(".")
    img = Image.open(gray_img)
    gray_img = np.array(img)

    path = os.getcwd() + "/tempDir/binary/"

    #plot histogram 
    plt.hist(gray_img, density = True, bins=25)  
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.title(" Histrogram of Pixels for given Image")
    plt.savefig(path + name + '_histogram' + "." + ext)

    grey_img = plt.imshow(gray_img,cmap='gray') 
    plt.axis('off')
    plt.title("Gray scale Image")
    plt.savefig(path + name + '_gray' + "." + ext)

    final_img = gray_img.copy()
    final_img[final_img > threshold] = 255
    final_img[final_img < threshold] = 0

    imgplotgrey = plt.imshow(final_img,cmap='gray') #gray scale
    plt.axis('off')
    plt.title(" Binary Threshold Image for given Threshold of " + str(threshold))
    plt.savefig(path + name + '_binary' + "." + ext)