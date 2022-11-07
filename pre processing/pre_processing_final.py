import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value



def pre_processing(data_path, outFileName):
    d = []  # list info images
    p = 0  # counter images

    max_delta_x = 0  # index for the max delta x
    max_delta_y = 0  # index for the max delta y


    for images in os.listdir(data_path):  # images in the folder
        path_image = os.path.join(data_path, images)
        if path_image.find('DS_Store') != -1:
            continue
        p = p + 1
        dict_mask = {}
        if p>300 and p<401:
                 # dict for ogni image

            print(p)
            T2_image = nib.load(path_image).get_fdata()  # load image

            mask = (T2_image[:, :, :] != 0)  # create a mask
            x = np.where(np.any(mask, axis=0))[0]  # value im !=0 y
            x_min, x_max = x[[0, -1]]  # max and min y

            y = np.where(np.any(mask, axis=1))[0]  # value im !=0 x
            y_min, y_max = y[[0, -1]]  # max and min x

            delta_x = x_max - x_min  # calculate dx
            delta_y = y_max - y_min  # calculate dy


            # dict with characteristich image
            dict_mask = {"num image": p,
                         "image": images,
                         "x": [x_min, x_max],
                         "y": [y_min, y_max],
                         "delta_x": delta_x,
                         "delta_y": delta_y,
                         "pixel_max": np.amax(T2_image),
                         "pixel_min": np.amin(T2_image)
                         }
            a=p
            d.append(dict_mask)  # final list


    f = open(outFileName, "w")
    f.write(str(a))
    f.write('\n')
    for elem in d:
        f.write(str(elem))
        f.write('\n')
        '''
        if elem['delta_x'] > max_delta_x:
            max_delta_x = elem["delta_x"]
            id_max_delta_x = elem["image"]
        if elem['delta_y'] > max_delta_y:
            max_delta_y = elem["delta_y"]
            id_max_delta_y = elem["image"]

        # crop image in the center
        x_star = int(max_delta_x/ 2)
        y_star = int(max_delta_y/2)


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

    for i in d:
        x_min_im = int(i['x'][0]) + int(i['delta_x'] / 2) - x_star
        x_max_im = int(i['x'][0]) + int(i['delta_x'] / 2) + x_star

        y_min_im = int(i['y'][0]) + int(i['delta_y'] / 2) - y_star
        y_max_im = int(i['y'][0]) + int(i['delta_y'] / 2) + y_star



        T2_image_final = i['im'][x_min_im:x_max_im, y_min_im:y_max_im,:]

        T2_image_final = np.pad(T2_image_final, 5, pad_with)

        T2_image_final = T2_image_final[:, :, :]
    f.write("Dimensioni immagine finale   ")
    f.write(str(np.shape(T2_image_final)))
    f.write('\n')
'''


data_path = "data/raw/T2_images"
outFileName = "./pre processing/pre_processing_T2_images/mask_dimension_T2_images_300_400.txt"


pre_processing(data_path,outFileName)





































#-----------------------------------------------------------------------------------------------------------------------


d = []  #list info images
p = 0   #counter images

max_delta_x = 0    #index for the max delta x
max_delta_y = 0    #index for the max delta y

for images in os.listdir(data_path):  # images in the folder
    dict_mask = {}    #dict for ogni image

    path_image = os.path.join(data_path, images)
    if path_image.find('DS_Store') != -1:
        continue
    p = p + 1
    if p < 100 :
        print(p)
        T2_image = nib.load(path_image).get_fdata()  # load image

        mask = (T2_image[:, :, :] != 0)    #create a mask
        y = np.where(np.any(mask, axis=0))[0]    #value im !=0 y
        y_min, y_max = y[[0, -1]]                #max and min y

        x = np.where(np.any(mask, axis=1))[0]    #value im !=0 x
        x_min, x_max = x[[0, -1]]                #max and min x


        delta_x = x_max - x_min                  #calculate dx
        delta_y = y_max - y_min                  #calculate dy

        #dict with characteristich image
        dict_mask = {"num image": p,
                     "image": images,
                     "x": [x_min, x_max],
                     "y": [y_min, y_max],
                     "delta_x": delta_x,
                     "delta_y": delta_y,
                     "pixel_max":  np.amax(T2_image),
                     "pixel_min": np.amin(T2_image),
                     "im": T2_image}

        d.append(dict_mask) #final list


outFileName = "./pre_processing_T2_images/mask_dimension_T2_images_prova.txt"
f = open(outFileName, "w")
for elem in d:
    #f.write(str(elem))
    #f.write('\n')
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
#f.write("delta_y: ")
#f.write(str(max_delta_y))
#f.write('\n')
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

    T2_image_final=np.pad(T2_image_final, 5, pad_with)

    T2_image_final = T2_image_final[:,:, 0:46]


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





outFileName = "./pre_processing_T2_images/mask_dimension_T2_images_800_888.txt"
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

    T2_image_final=np.pad(T2_image_final, 5, pad_with)

    T2_image_final = T2_image_final[:,:, 0:46]


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



outFileName = "./pre_processing_T2_images/mask_dimension_T2_images_800_888.txt"
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

    T2_image_final=np.pad(T2_image_final, 5, pad_with)

    T2_image_final = T2_image_final[:,:, 0:46]


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



#-------------------------------------
outFileName = "./pre_processing_T2_images/mask_dimension_T2_images_800_888.txt"
f = open(outFileName, "w")
for elem in d:

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

    T2_image_final=np.pad(T2_image_final, 5, pad_with)

    T2_image_final = T2_image_final[:,:, 0:46]

    f.write("----------------------------------------------------------------------------------------------------------------------")
f.close()





#------------------------------------------------Funzione finale-------------------------------------------------------
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


