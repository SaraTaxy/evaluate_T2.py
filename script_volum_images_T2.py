def crop_center(data, out_sp):
    """
    crop_center --> dal centro verso tutte le altre dimensioni
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example:
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)                      #dimensione output finale
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)  #x,y,z =3 almeno dovrebbe essere
    x_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    #z_crop = int((in_sp[-3] - out_sp[-1]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, out_sp[-1]:-out_sp[-1]]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, out_sp[-1]:-out_sp[-1]]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop


def format_image_T2(data):
    image = np.rot90(data)
    nd = np.ndim(data)    #dimension
    shape = data.shape
    val_min = np.amin(data)
    val_max = np.amax(data)

    return image, nd, shape, val_min, val_max



import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

T2_image = nib.load('data/raw/T2_images/RecordID_0015_T2_flair.nii').get_fdata()
T2_image = np.rot90(T2_image)
nd = np.ndim(T2_image)
print(nd)
print(T2_image.shape)



#print 1 slice
layer=27
plot_T2 = T2_image[:,:,layer]
plt.imshow(plot_T2)
plt.gcf().set_size_inches(5, 5)
plt.style.use('grayscale')
plt.show()



#print all slice

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(T2_image[:,:, 10+i])
    plt.gcf().set_size_inches(5,5)
    plt.style.use('grayscale')
plt.show()

#find value pixel max/min

max_pixel = np.amax(T2_image)
print(max_pixel)

min_pixel = np.amin(T2_image)
print(min_pixel)

#crop  --> sbagliato perè do le dimensioni in ingresso

crop_T2 = crop_center(T2_image,(325,250,1))
plt.imshow(crop_T2[:,:, 33])

#create a mask

mask = (T2_image[:,:,layer])>0
plt.subplot(1,2,1)
plt.imshow(T2_image[:,:,layer])
plt.title("Immagine iniziale")
plt.subplot(1,2,2)
plt.imshow(mask)
plt.title("mask")

print(mask.shape)

#xy
i_list_xy=[]
j_list_xy=[]
for i in range(0, mask.shape[0]):
    for j in range(0, mask.shape[1]):
        if mask[i][j]==True:
            #print(i, j)
            i_list_xy.append(i)
            j_list_xy.append(j)

p_max_xy = []
p_max_xy.append(i_list_xy[-1])
#p_max_xy.append(j_list_xy[-1])

print(p_max_xy)

p_min_xy = []
p_min_xy.append(i_list_xy[0])
#p_min_xy.append(j_list_xy[0])

print(p_min_xy)

#yx
mask=np.transpose(mask)

i_list_yx=[]
j_list_yx=[]

for i in range(0, mask.shape[0]):
    for j in range(0, mask.shape[1]):
        if mask[i][j]==True:
            #print(i,j)
            i_list_yx.append(i)
            j_list_yx.append(j)



p_max_yx = []
p_max_yx.append(i_list_yx[-1])
#p_max_xy.append(j_list_xy[-1])

print(p_max_xy)

p_min_yx = []
p_min_yx.append(i_list_yx[0])
#p_min_xy.append(j_list_xy[0])

print(p_min_yx)


print("y min: ", p_min_yx, "y max: ", p_max_yx)
print("x min: ", p_min_xy, "x max: ", p_max_xy)

dim_mask = {}

dim_mask = {"immagine": 1, "nome immagine": "data/raw/T2_images/RecordID_0012_T2_flair.nii", "dimensione mask": {"x":[p_min_xy[0],p_max_xy[0]],"y": [p_min_yx[0],p_max_yx[0]]}}


T2_image_cropped=T2_image[p_min_xy[0]:p_max_xy[0],p_min_yx[0]:p_max_yx[0], layer]
mask=np.transpose(mask)

plt.subplot(1,3,1)
plt.imshow(T2_image[:,:,layer])
plt.title("Immagine iniziale")
plt.subplot(1,3,2)
plt.imshow(mask)
plt.title("mask")
plt.subplot(1,3,3)
plt.imshow(T2_image_cropped)
plt.title("immagine tagliata")


mask_3D = (T2_image[:,:,:])>0
print(mask_3D.shape)

z1 =[]
for i in range(0, mask_3D.shape[0]):
    for j in range(0, mask_3D.shape[1]):
        for z in range(0, mask_3D.shape[2]):
            if mask_3D[i][j][z]==True:
                z1.append(z)
                #print(i,j,z)
print(np.amax(z))
            #i_list_yx.append(i)
            #j_list_yx.append(j)



#----------------------------------------for all images -----------------------------------------------------------------------

#provato--> funziona ma problemi che scambia y e z --> credo che un problema sia la trasposta
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np


data_path = "data/raw/T2_images"

dict_mask = {}
for images in os.listdir(data_path):
    path_image = os.path.join(data_path, images)
    if path_image.find('DS_Store') != -1:
        continue

    T2_image = nib.load(path_image).get_fdata()
    T2_image = np.rot90(T2_image)
    nd = np.ndim(T2_image)
    max_pixel = np.amax(T2_image)
    min_pixel = np.amin(T2_image)

    mask = (T2_image[:, :, :]) > 0

    i_list_xy = []
    j_list_xy = []

    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            for z in range(0, mask.shape[2]):
                if mask[i][j][z] == True:
                # print(i, j)
                    i_list_xy.append(i)
                    j_list_xy.append(j)


    p_max_xy = []
    p_max_xy.append(i_list_xy[-1])
    # p_max_xy.append(j_list_xy[-1])

    #print(p_max_xy)

    p_min_xy = []
    p_min_xy.append(i_list_xy[0])
    # p_min_xy.append(j_list_xy[0])

    #print(p_min_xy)

    # yx
    mask = np.transpose(mask)

    i_list_yx = []
    j_list_yx = []

    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            for z in range(0, mask.shape[2]):
                if mask[i][j][z] == True:
                    # print(i, j)
                    i_list_yx.append(i)
                    j_list_yx.append(j)

    p_max_yx = []
    p_max_yx.append(i_list_yx[-1])
    # p_max_xy.append(j_list_xy[-1])

    print(p_max_xy)

    p_min_yx = []
    p_min_yx.append(i_list_yx[0])
    # p_min_xy.append(j_list_xy[0])

    print(p_min_yx)

    print("y min: ", p_min_yx, "y max: ", p_max_yx)
    print("x min: ", p_min_xy, "x max: ", p_max_xy)


    dict_mask = {"immagine": 1, "nome immagine": images , "dimensione mask": {"x": [p_min_xy[0], p_max_xy[0]], "y": [p_min_yx[0], p_max_yx[0]], "z": [0,46]}}



for images in os.listdir(data_path):
    path_image = os.path.join(data_path, images)
    print(path_image)
    #T2_image = nib.load(path_image).get_fdata()





#---------------------------------Questo è correct------------------------------------------------------------------------------
#prova 2

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


data_path = "data/raw/T2_images"

d=[]
p=0

#dict_mask = {}
for images in os.listdir(data_path):
    dict_mask = {}


    if p<=10:
        p = p + 1
        print(p)
        path_image = os.path.join(data_path, images)
        if path_image.find('DS_Store') != -1:
            continue

        T2_image = nib.load(path_image).get_fdata()
        T2_image = np.rot90(T2_image)
        nd = np.ndim(T2_image)
        max_pixel = np.amax(T2_image)
        min_pixel = np.amin(T2_image)

        mask = ((T2_image[:,:,:])>0)

        i_list = []
        j_list = []
        z_list = []

        for z in range(0, mask.shape[2]):
            for i in range(0, mask.shape[0]):
                for j in range(0, mask.shape[1]):
                    if mask[i][j][z]==True:
                        j_list.append(j)
                        i_list.append(i)
                        z_list.append(z)


        '''
        print("nome immagine: ", path_image)
        print("x min: ", np.amin(i_list), "x max: ", np.amax(i_list))
        print("y min: ", np.amin(j_list), "y max: ", np.amax(j_list))
        print("z min: ", np.amin(z_list), "z max: ", np.amax(z_list))
        '''
        vol = (np.amax(i_list)-np.amin(i_list))*(np.amax(j_list)-np.amin(j_list))*(np.amax(z_list)-np.amin(z_list))

        #print("vol in pixel: ", vol)
        dict_mask = {"num image": p,
                     "image": images,
                     "x": [np.amin(i_list), np.amax(i_list)],
                     "y": [np.amin(j_list), np.amax(j_list)],
                     "z": [np.amin(z_list), np.amax(z_list)],
                     "Pixel_Vol": vol,
                     "delta_x": (np.amax(i_list) - np.amin(i_list)),
                     "delta_y": (np.amax(j_list) - np.amin(j_list)),
                     "delta_z": (np.amax(z_list) - np.amin(z_list))}
        d.append(dict_mask)
    else:
        p = p + 1

print(d)


max_x = 0
index_x = 0
max_y = 0
index_y = 0
max_z = 0
index_z = 0
for i in range(0, len(d)):
    if d[i]['delta_x'] > max_x:
        max_x = d[i]['x'][1] - d[i]['x'][0]
        im_x=d[i]['image']
        index_x = i
    else:
        max_x = max_x

    if d[i]['delta_y'] > max_y:
        max_y = d[i]['y'][1] - d[i]['y'][0]
        im_y=d[i]['image']
        index_y = i
    else:
        max_y = max_y

    if d[i]['delta_z'] > max_z:
        max_z = d[i]['z'][1] - d[i]['z'][0]
        im_z=d[i]['image']
        index_z = i
    else:
        max_z = max_z



#crop image in the center
x_star = int(max_x/2)
y_star = int(max_y/2)
z_star = int(max_z/2)


for i in range(0, len(d)):

    x_min_im = int(d[i]['x'][0]) + int(d[i]['delta_x']/2) - x_star
    x_max_im = int(d[i]['x'][0]) + int(d[i]['delta_x']/2) + x_star
    #print("delta_x im ", i)
    #print([x_min_im, x_max_im])

    y_min_im = int(d[i]['y'][0]) + int(d[i]['delta_y'] / 2) - y_star
    y_max_im = int(d[i]['y'][0]) + int(d[i]['delta_y'] / 2) + y_star

    z_min_im = int(d[i]['z'][0]) + int(d[i]['delta_z'] / 2) - z_star
    z_max_im = int(d[i]['z'][0]) +int(d[i]['delta_z'] / 2) + z_star



# create file
with open("mask_dimension_T2_images.txt", 'w') as f:
    for elem in d:
        f.write(str(elem))
        f.write('\n')
    f.write('\n')

    # find max value of images volum
    f.write("----------------------------------------DIM MASK MAX VOLUM IMAGE-----------------------------------------------------")
    f.write('\n')
    f.write("volum max: ")
    f.write(str(max))
    f.write('\n')
    f.write("index image x max: ")
    f.write(str(im_x))
    f.write('\n')
    f.write("delta_x: ")
    f.write(str(max_x))
    f.write('\n')
    f.write("index image y max: ")
    f.write(str(im_y))
    f.write('\n')
    f.write("delta_y: ")
    f.write(str(max_y))
    f.write('\n')
    f.write("index image z max: ")
    f.write(str(im_z))
    f.write('\n')
    f.write("delta_z: ")
    f.write(str(max_z))
    f.write('\n')

    # crop image in the center
    x_star = int(max_x/2)
    y_star = int(max_y/2)
    z_star = int(max_z/2)

    f.write("----------------------------------------END MASK FOR EACH IMAGES-----------------------------------------------------")
    f.write('\n')

    for i in range(0, len(d)):
        x_min_im = int(d[i]['x'][0]) + int(d[i]['delta_x'] / 2) - x_star
        x_max_im = int(d[i]['x'][0]) + int(d[i]['delta_x'] / 2) + x_star
        # print("delta_x im ", i)
        # print([x_min_im, x_max_im])

        y_min_im = int(d[i]['y'][0]) +int(d[i]['delta_y'] / 2) - y_star
        y_max_im = int(d[i]['y'][0]) + int(d[i]['delta_y'] / 2) + y_star

        z_min_im = int(d[i]['z'][0]) + int(d[i]['delta_z'] / 2) - z_star
        z_max_im = int(d[i]['z'][0]) + int(d[i]['delta_z'] / 2) + z_star


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
        f.write("z_min: ")
        f.write(str(z_min_im))
        f.write(" z_max: ")
        f.write(str(z_max_im))
        f.write('\n')
        f.write("------------------------------")
        f.write('\n')


with open("mask_dimension_T2_images.txt", 'w') as f:
    for elem in d:
        f.write(str(elem))
        f.write('\n')
    f.write('\n')
    f.write("----------------------------------------DIM MASK MAX VOLUM IMAGE-----------------------------------------------------")
    # find max value of images volum
    max=0
    index=0


    for i in range(0, len(d)):
        if d[i]['Pixel_Vol']> max:
            max=d[i]['Pixel_Vol']
            index=i
        else:
            max=max

    f.write('\n')
    f.write("volum max: ")
    f.write(str(max))
    f.write('\n')
    f.write("index image vol max: ")
    f.write(str(d[index]['image']))
    f.write('\n')
    f.write("x: ")
    f.write(str(d[index]['x']))
    f.write('\n')
    f.write("y: ")
    f.write(str(d[index]['y']))
    f.write('\n')
    f.write("z: ")
    f.write(str(d[index]['z']))
    f.write('\n')






#-----------------save cvs file ---------------
d={}
j=0
for i in range(0,10):
    d_n={}
    j=+1
    n=10*i
    d_n={'nome': i, "valore": n}
    d[i]=d_n

print(d)

with open("mask_dimension_T2_images.txt", 'w') as f:
    for key, value in d.items():
        f.write('%s:%s\n' % (key, value))



#--------------max  in dict

import numpy as np

d={}
j=0
l = []
nome = []
for i in range(0,10):
    d_n={}
    j=+1
    n=10*i
    d_n={'nome': i, "valore": n}
    d[i]=d_n
    l.append(d_n['valore'])
    nome.append(d_n['nome'])

print(nome)

max=0
i=0
for q in range(0, len(l)):
    if l[q]>max:
        max=l[q]
        i=q
    else:
        max=max

print(nome[i], max)
