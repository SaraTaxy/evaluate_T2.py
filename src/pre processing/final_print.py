
import nibabel as nib
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def cropping_T2(img, delta_x_max, delta_y_max):


    mask = (img[:, :, :] != 0)  # create a mask
    x = np.where(np.any(mask, axis=0))[0]  # value im !=0 y
    x_min, x_max = x[[0, -1]]  # max and min y
    #print(x_min, x_max)

    y = np.where(np.any(mask, axis=1))[0]  # value im !=0 x
    y_min, y_max = y[[0, -1]]  # max and min x
    #print(y_min, y_max)

    delta_x = x_max - x_min  # calculate dx
    delta_y = y_max - y_min

    x_star = round(delta_x_max/2)                # required to have the same size as parallelepiped along x --> delta_x_max/2
    y_star = round(delta_y_max/2)                # required to have the same size as parallelepiped along y --> delta_y_max/2

    delta_x_imag = int((x_max - x_min)/2)  # find delta_x image i-esim
    delta_y_imag = int((y_max - y_min)/2)  # find delta_y image i-esim


    x_min_im = (x_min + delta_x_imag) - x_star   # find the x min for the new image cropped

    x_max_im = (x_min + delta_x_imag) + x_star # find the x max for the new image cropped
    print(x_min_im, x_max_im)


    y_min_im = (y_min + delta_y_imag) - y_star  # find the y min for the new image cropped

    y_max_im = (y_min + delta_y_imag) + y_star       # find the y max for the new image cropped
    print(y_min_im, y_max_im)

    # risistema le variabili

    T2_image_final_1 = img[y_min_im:y_max_im,x_min_im:x_max_im , :]  # image final

    T2_image_final = np.pad(T2_image_final_1, 5, pad_with)           # zero padding --> addition of null pixels to the image

    print(T2_image_final.shape)
    #print(T2_image_cropped.shape)

    return mask, T2_image_final_1, T2_image_final


T2_image = nib.load('data/raw/T2_images/RecordID_2120_T2_flair.nii').get_fdata()



mask, T2_image_final_1, T2_cropped = cropping_T2(T2_image, 396, 331)

layer=25

plt.subplot(1,4,1)
plt.style.use('grayscale')
plt.imshow((T2_image[:,:,layer]))
plt.title("iniziale")

plt.subplot(1,4,2)
plt.style.use('grayscale')
plt.imshow(mask[:,:,layer])
plt.title("mask")


plt.subplot(1,4,3)
plt.style.use('grayscale')
plt.imshow(T2_image_final_1[:,:,layer])
plt.title("Immagine tagliata")


plt.subplot(1,4,4)
plt.style.use('grayscale')
plt.imshow(T2_cropped[:,:,layer+5])
plt.title("zero pad")


a = np.zeros(shape=(342,406,22))
b = np.concatenate((a, T2_cropped), axis=2)


# ----
zeros = np.zeros(shape=(342,406,22))
new_image = np.dstack((zeros, T2_cropped))
new_image = np.dstack((new_image, zeros))


print(c.shape)



plt.subplot(1,2,1)
plt.style.use('grayscale')
plt.imshow(T2_cropped[:,:,47])
plt.title("zero pad")

plt.subplot(1,2,2)
plt.imshow(new_image[:,:,47])
plt.title("new")
