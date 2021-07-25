# importing necessary packages

import cv2
import numpy as np
import tifffile as tif
import tensorflow as tf


#  function to read the RGB Images using open-cv library

def read_ldim(image_path):
    img = cv2.imread(image_path)           # reading the RGB image using the provided image path
    height = img.shape[0]                  # taking the height of the images as the number of rows
    width = img.shape[1]                   # taking the width of the images as the number of columns
    return img,height,width                             # returning the resultant image array


#  function to read the tif DEM and Landslide images using tifffile library

def read_tifim(image_path,height,width):  
    img = tif.imread(image_path)             # reading the tif file using the tifffile library
    if(height!=img.shape[0] or width!= img.shape[1]):
      img = cv2.resize(img,(width,height))     # resizing the image to be of the same height and width as of the GEO image
    img = img.reshape(img.shape[0],img.shape[1],1)        # reshaping the image resulting in 1 channel
    return img   

                            # returning the resultant image array
                        
                        
# Label 0  stands for NON LANDSLIDE
# LabeL 1 stands for LANDSLIDES
# Label 3 is returned for the area which is not in the region of interest. We do not know the landslide region in this area so we do not need the images with label 3 in the data that we are creating.

# Black area stands for the non landslide parts of the data so the label returned is 0.
# White area stands for the region which is not in the region of interest so label returned is 3 and these images are not saved.
# Great area stand for the region in which the landslides occur so the label returned is 1.
# For partial black and white area we take the label as 0.
# For partial black and partial grey area we take the label as 1.

# function to assign the labels using the landslide image color codes
def assign_label(ld_crop):
    unique = np.unique(ld_crop)                              # taking all the unique color codes in the cropped landslide image
    l = len(unique)                                          # the length of the list of different color codes is stored in variable l
    if l == 1 and unique[0]==255:                            #total white area
        return 3
    elif l == 1 and unique[0]==0:                            #total black area
        return 0
    elif l == 2 and unique[0] == 0 and unique[1] == 255:     #partial white and black area
        return 0 
    elif l == 2 and unique[0] == 0 and unique[1] != 255:     #partial black and partial grey area
        return 1
    else:                                                    #total grey area
        return 1 


# more about tf.image.crop_to_bounding_box --> https://www.tensorflow.org/api_docs/python/tf/image/crop_to_bounding_box\

# offset_height - Vertical coordinate of the top-left corner of the result in the input.
# offset_width  - Horizontal coordinate of the top-left corner of the result in the input.
# target_height - Height of the result.
# target_width  - Width of the result.

# we iterate over the whole RGB and the whole DEM  image cutting 16x16 pixel images which are mutually exclusive ending the loop if the width exceeds the width of the total image.

# all the images cutted are mutually exclusive and do not coincide with each other.

# this function is used for creating sliding windows on the images of height and width 16 pixels.
def split(image,dem,ld):
    offset_height = 0               
    target_height = 16
    target_width = 16
    t = 0                        # t is count of number of images that are saved in the memory which is also printed later in the code.
    m = int(image.shape[0]/16)   # m stroes the number of iterations over the rows which is equal to the height of the imgae(the number of rows) divided by the height of the each image in pixels considering all the images cut are mutually exclusive.
    r = int(image.shape[1]/16)   # r stroes the number of iterations over the columns which is equal to the width of the imgae(the number of columns) divided by the width of the each image in pixels considering all the images cut are mutually exclusive.
    for i in range(m):           # iterating over the rows
        offset_width = 0
        for j in range(r):       # iterating over the columns
            img = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)   # cropping 16x16 RGB images from the GEO image
            dm = tf.image.crop_to_bounding_box(dem, offset_height, offset_width, target_height, target_width)      # cropping 16x16 DEM images from the DEM image corresponding to the same area.
            ld_crop = tf.image.crop_to_bounding_box(ld,offset_height,offset_width,target_height,target_width)      # cropping 16x16 LANDSLIDE images from the DEM image corresponding to the same area.
            im1 = np.concatenate((img,dm),axis=2)                                                                  # concatenating the RGB and the DEM image for the same area to make the resultant image to be 4 channels.
            label = assign_label(ld_crop)                                                                          # assigning labels using the assign_labels function defined above in the code. 
            if label != 3:                                                                                         # if label is 3  the image is not save din the memory else it is saved.
                tif.imsave(f'C:/Users/muska/Desktop/saved_geo8/GEO8_{t}_{label}.tif',im1)                                                 # saving the image in the memory using tifffile imsave
                t = t + 1                                                                                          # increasing the saved images count by 1
            offset_width = offset_width + 16                                                                       # increaing offset width by 16 to cover the next mutually exclusive image.
            if offset_width > image.shape[1]:                                                                      # if in case the offset width exceeds the total width of the image the loop breaks
                break
        print('Saved : ',t)                               # printing the number of images saved in the process.
        offset_height = offset_height + 16                # increasing offset height by 16.
        if offset_height > image.shape[0]:                # if in case the offset height exceeds the total height of the image the loop breaks
            break


im,height,width = read_ldim('C:/Users/muska/Desktop/geo_8/GEO_8_N.tif')    # reading the whole GEO image using the read_ldim function                                                                          # taking width of the image using number of columns
dem = read_tifim('C:/Users/muska/Desktop/geo_8/dem_8_N.tif',height,width)                                            # reading the DEM image for the corresponding area of the GEO image using read_tifim function
ld = read_tifim('C:/Users/muska/Desktop/geo_8/landslide_8_N.tif',height,width)   


split(im,dem,ld)                             # reading the LANDSLIDE image for the corresponding area of the GEO image using read_tifim function