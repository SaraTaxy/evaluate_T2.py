import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd



def pre_processing_T2(data_path):
    d = []  #list info images
    p = 0   #counter images

    #for delta_x image max and index image
    max_x = 0
    index_x = 0

    # for delta_y image max and index image
    max_y = 0
    index_y = 0

    o = []          #list value max and min pixel
    T2_total = []   #list of array of images with pixel !=0   --> array([], [])

    final_hist = []    #list of value pixel  != 0   --> need to histogram

    for images in os.listdir(data_path):                        #images in the folder
        dict_mask = {}
        pixel = {}


        path_image = os.path.join(data_path, images)
        if path_image.find('DS_Store') != -1:
            continue
        p = p + 1
        if p<=2:
            T2_image = nib.load(path_image).get_fdata()    #load image
            T2_image = np.rot90(T2_image)                  #rotation image
            nd = np.ndim(T2_image)

            max_pixel = np.amax(T2_image)                 #find pixel max
            min_pixel = np.amin(T2_image)                 #find pixel min

            mask = ((T2_image[:, :, :]) > 0)              #create a mask

            i_list = []
            j_list = []


            for z in range(0, mask.shape[2]):
                for i in range(0, mask.shape[0]):
                    for j in range(0, mask.shape[1]):
                        if mask[i][j][z] == True:
                            j_list.append(j)
                            i_list.append(i)

            max_x = np.amax(i_list)
            max_y = np.amax(j_list)


            min_x = np.amin(i_list)
            min_y = np.amin(j_list)


            T2_image_cropped = T2_image[min_x:max_x, min_x:max_x, 0:46]

            T2_image_final = 0

            T2_total.append(T2_image_cropped[T2_image_cropped != 0])

            dict_mask = {"num image": p,
                         "image": images,
                         "x": [np.amin(i_list), np.amax(i_list)],
                         "y": [np.amin(j_list), np.amax(j_list)],
                         "delta_x": (np.amax(i_list) - np.amin(i_list)),
                         "delta_y": (np.amax(j_list) - np.amin(j_list)),
                         "dim image begin": np.shape(T2_image),
                         "dim image cropped": np.shape(T2_image_cropped),
                         "dim image +5 pixel x and y": T2_image_final,
                         "imag": T2_image}


            d.append(dict_mask)

            pixel = {"image": images,
                     "value pixel max": max_pixel,
                     "value pixel min": min_pixel}
            o.append(pixel)

            #p = p + 1
            print(p)

            '''
            for elem in T2_total:
                for l in elem:
                    final_hist.append(l)
            '''
        max_x = 0
        index_x = 0
        max_y = 0
        index_y = 0
        max_z = 0
        index_z = 0
        for i in range(0, len(d)):
            if d[i]['delta_x'] > max_x:
                max_x = d[i]['x'][1] - d[i]['x'][0]
                im_x = d[i]['image']
                index_x = i
            else:
                max_x = max_x

            if d[i]['delta_y'] > max_y:
                max_y = d[i]['y'][1] - d[i]['y'][0]
                im_y = d[i]['image']
                index_y = i
            else:
                max_y = max_y

    return d, im_x, max_x, im_y, max_y, final_hist, o


data = "data/raw/T2_images"
d, im_x, max_x, im_y, max_y, final_hist, o = pre_processing_T2(data)




#file info mask images
outFileName = "./pre_processing_T2_images/mask_dimension_T2_images_prova.txt"
f = open(outFileName, "w")
for elem in d:
    f.write(str(elem))
    f.write('\n')
f.write('\n')
# find max value of images volum
f.write("----------------------------------------DIM MASK MAX VOLUM IMAGE-----------------------------------------------------")
f.write('\n')
#f.write("index image x max: ")
#f.write(str(im_x))
f.write('\n')
f.write("delta_x: ")
f.write(str(max_x))
f.write('\n')
#f.write("index image y max: ")
#f.write(str(im_y))
f.write('\n')
f.write("delta_y: ")
f.write(str(max_y))
f.write('\n')
# crop image in the center
x_star = int(max_x / 2)
y_star = int(max_y / 2)

f.write("----------------------------------------END MASK FOR EACH IMAGES-----------------------------------------------------")

for i in range(0, len(d)):
    x_min_im = int(d[i]['x'][0]) + int(d[i]['delta_x'] / 2) - x_star
    x_max_im = int(d[i]['x'][0]) + int(d[i]['delta_x'] / 2) + x_star
    # print("delta_x im ", i)
    # print([x_min_im, x_max_im])

    y_min_im = int(d[i]['y'][0]) + int(d[i]['delta_y'] / 2) - y_star
    y_max_im = int(d[i]['y'][0]) + int(d[i]['delta_y'] / 2) + y_star

    T2_image_final = d[i]['imag'][x_min_im:x_max_im, y_min_im:y_max_im, :]

    dim1 = np.zeros((5, np.shape(T2_image_final)[1], np.shape(T2_image_final)[2]))

    T2_image_final = np.vstack((dim1, T2_image_final))
    T2_image_final = np.vstack((T2_image_final, dim1))
    dim2 = np.zeros((np.shape(T2_image_final)[0], 5, np.shape(T2_image_final)[2]))
    T2_image_final = np.hstack((dim2, T2_image_final))
    T2_image_final = np.hstack((T2_image_final, dim2))

    f.write('\n')
    f.write("x_min: ")
    f.write(str(x_min_im))
    f.write(" x_max: ")
    f.write(str(x_max_im))
    f.write('\n')
    f.write("y_min: ")
    f.write(str(y_min_im))
    f.write(" y_max: ")
    f.write(str(y_max_im))
    f.write('\n')
    f.write("final image: ")
    f.write(str(np.shape(T2_image_final)))
    f.write('\n')
    f.write("x_min_new: ")
    f.write(str(x_min_im - 5))
    f.write(" x_max_new: ")
    f.write(str(x_max_im + 5))
    f.write('\n')
    f.write("y_min_new: ")
    f.write(str(y_min_im - 5))
    f.write(" y_max_new: ")
    f.write(str(y_max_im + 5))
    f.write('\n')
    f.write("----------------------------------------------------------------------------------------------------------------------")
f.close()

#file info pixel images
outFile = "./pre_processing_T2_images_50_100/Pixel.txt"
g = open(outFile, "w")
for elem in o:
    g.write(str(elem))
    g.write('\n')
g.write('\n')
g.close()


#save histogram
plt.hist(final_hist)
plt.savefig('./pre_processing_T2_images/instogram_t2_final')




#----------------------------------- MAX AND MIN PIXEL IMAGES ----------------------------------------------------------
def max_min_pixel(data_path):
    max_im = 0
    min_im = 0
    #index_max_pixel = ''
    d = []  # list info images
    p = 0
    max_pixel = 0
    min_pixel = 0

    for images in os.listdir(data_path):
        dict_mask = {}


        path_image = os.path.join(data_path, images)
        if path_image.find('DS_Store') != -1:
            continue
        p = p + 1
        print(p)
        T2_image = nib.load(path_image).get_fdata()
        max_pixel = np.amax(T2_image)
        min_pixel = np.amin(T2_image)

        dict_mask = {"image": images,
                     "max_pixel": max_pixel,
                     "min pixel": min_pixel}

        d.append(dict_mask)

        if dict_mask["max_pixel"] > max_im:
            max_im = dict_mask["max_pixel"]
            index_max_pixel = dict_mask["image"]
        else:
            max_pixel = max_pixel

    outFile = "./pre_processing_T2_images/Pixel.txt"
    g = open(outFile, "w")
    for elem in d:
        g.write(str(elem))
        g.write('\n')
    g.write('\n')
    g.write("---------")
    g.write('\n')
    g.write("max pixel: ")
    g.write('\n')
    g.write(str(max_im))
    g.write('\n')
    g.write("max pixel index image: ")
    g.write('\n')
    g.write(str(index_max_pixel))
    g.close()


data = "data/raw/T2_images"
max_min_pixel(data)


#-----------------------------------------------------new correct ------------------------------------------------------

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os


data_path = "data/raw/T2_images"

d = []  #list info images
p = 0   #counter images

max_delta_x = 0
max_delta_y = 0

for images in os.listdir(data_path):  # images in the folder
    dict_mask = {}

    path_image = os.path.join(data_path, images)
    if path_image.find('DS_Store') != -1:
        continue
    p = p + 1
    if p <= 5:
        print(p)
        T2_image = nib.load(path_image).get_fdata()  # load image

        mask = (T2_image[:, :, :] != 0)
        y = np.where(np.any(mask, axis=0))[0]
        y_min, y_max = y[[0, -1]]

        x = np.where(np.any(mask, axis=1))[0]
        x_min, x_max = x[[0, -1]]


        delta_x = x_max - x_min
        delta_y = y_max - y_min


        dict_mask = {"num image": p,
                     "image": images,
                     "x": [x_min, x_max],
                     "y": [y_min, y_max],
                     "delta_x": delta_x,
                     "delta_y": delta_y,
                     "pixel_max":  np.amax(T2_image),
                     "pixel_min": np.amin(T2_image),
                     "im": T2_image}

        d.append(dict_mask)


outFileName = "./pre_processing_T2_images/mask_dimension_T2_images_prova.txt"
f = open(outFileName, "w")
for elem in d:

    f.write(str(elem))
    f.write('\n')
    if elem['delta_x']> max_delta_x:
        max_delta_x = elem["delta_x"]
        id_max_delta_x = elem["image"]
    if elem['delta_y']> max_delta_y:
        max_delta_y = elem["delta_y"]
        id_max_delta_y = elem["image"]
f.write('\n')
f.write("----------------------------------------DIM MASK MAX VOLUM IMAGE-----------------------------------------------------")
f.write('\n')
f.write("index image x max: ")
f.write(str(id_max_delta_x))
f.write('\n')
f.write("delta_x: ")
f.write(str(max_delta_x))
f.write('\n')
f.write("index image y max: ")
f.write(str(id_max_delta_y))
f.write('\n')
f.write("delta_y: ")
f.write(str(max_delta_y))
f.write('\n')
# crop image in the center
x_star = int(max_delta_x / 2)
y_star = int(max_delta_y / 2)

f.write("----------------------------------------END MASK FOR EACH IMAGES-----------------------------------------------------")



for i in d:
    x_min_im = int(i['x'][0]) + int(i['delta_x'] / 2) - x_star
    x_max_im = int(i['x'][0]) + int(i['delta_x'] / 2) + x_star

    y_min_im = int(i['y'][0]) + int(i['delta_y'] / 2) - y_star
    y_max_im = int(i['y'][0]) + int(i['delta_y'] / 2) + y_star


    T2_image_final = i['im'][x_min_im:x_max_im, y_min_im:y_max_im, :]

    dim1 = np.zeros((5, np.shape(T2_image_final)[1], np.shape(T2_image_final)[2]))

    T2_image_final = np.vstack((dim1, T2_image_final))
    T2_image_final = np.vstack((T2_image_final, dim1))

    dim2 = np.zeros((np.shape(T2_image_final)[0], 5, np.shape(T2_image_final)[2]))
    T2_image_final = np.hstack((dim2, T2_image_final))
    T2_image_final = np.hstack((T2_image_final, dim2))


    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value


    a = np.arange(6)
    a = a.reshape((2, 3))
    np.pad(a, 2, pad_with)


    f.write('\n')
    f.write("x_min start: ")
    f.write(str(x_min_im))
    f.write("   x_max start: ")
    f.write(str(x_max_im))
    f.write('\n')
    f.write("y_min start: ")
    f.write(str(y_min_im))
    f.write("   y_max start: ")
    f.write(str(y_max_im))
    f.write('\n')
    f.write("final image: ")
    f.write(str(np.shape(T2_image_final)))
    f.write('\n')
    f.write("x_min_new: ")
    f.write(str(x_min_im - 5))
    f.write("   x_max_new: ")
    f.write(str(x_max_im + 5))
    f.write('\n')
    f.write("y_min_new: ")
    f.write(str(y_min_im - 5))
    f.write("   y_max_new: ")
    f.write(str(y_max_im + 5))
    f.write('\n')
    f.write("----------------------------------------------------------------------------------------------------------------------")
f.close()


#-----------------------------------------------------------------------------------------------------------------------




import torch
import os
import random
from scipy.ndimage import shift
import nibabel as nib
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt





def load_img(img_path):
    img = nib.load(img_path)
    img = img.get_fdata()
    return img


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def cropping_T2(img, delta_x, delta_y):

    mask = (img[:, :, :] != 0)  # create a mask
    y = np.where(np.any(mask, axis=0))[0]  # value im !=0 y
    y_min, y_max = y[[0, -1]]  # max and min y

    x = np.where(np.any(mask, axis=1))[0]  # value im !=0 x
    x_min, x_max = x[[0, -1]]  # max and min x

    x_star = int(delta_x/2)
    y_star = int(delta_y/2)

    delta_x_imag = int((x_max - x_min)/2)
    delta_y_imag = int((y_max - y_min)/2)

    x_min_im = x_min + delta_x_imag - x_star - 1
    x_max_im = x_min + delta_x_imag + x_star

    y_min_im = y_min + delta_y_imag - y_star
    y_max_im = y_min + delta_y_imag + y_star


    T2_image_final = img[x_min_im:x_max_im, y_min_im:y_max_im, :]
    T2_image_final = np.pad(T2_image_final, 5, pad_with)
    T2_image_cropped = T2_image_final[:, :, 0:47]
    print(T2_image_cropped.shape)

    return T2_image_cropped


img = load_img('data/raw/T2_images/RecordID_4838_T2_flair.nii')



img = nib.load('data/raw/T2_images/RecordID_0015_T2_flair.nii').get_fdata()
img = cropping_T2(img, 325, 396)


