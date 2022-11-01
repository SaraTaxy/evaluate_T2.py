
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

T2_image = nib.load('data/raw/T2_images/RecordID_0015_T2_flair.nii').get_fdata()
mask = (T2_image[:,:,:] != 0)
y = np.where(np.any(mask, axis=0))[0]
y_min, y_max = y[[0, -1]]

x = np.where(np.any(mask, axis=1))[0]
x_min, x_max = x[[0, -1]]


plt.imshow(T2_image[x_min:x_max, y_min:y_max, 12])
plt.show()

max_delta_x = 0
max_delta_y = 0

delta_x = x_max -x_min
delta_y = y_max -y_min


if delta_x > max_delta_x:
    max_delta_x = delta_x
    #salva l'indice dell'immagine

if delta_y > max_delta_y:
    max_delta_y = delta_y
    #salva l'indice dell'immagine

print(max_delta_x)
print(max_delta_y)








import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


T2_image = nib.load('data/raw/T2_images/RecordID_0015_T2_flair.nii').get_fdata()
T2_image = np.rot90(T2_image)
nd = np.ndim(T2_image)
print(nd)
print(T2_image.shape)

a= []
l = []
k =[]


T2_image_1 = nib.load('data/raw/T2_images/RecordID_0012_T2_flair.nii').get_fdata()
T2_image_1 = np.rot90(T2_image_1)
T2 = np.hstack((T2_image, T2_image_1))

T2_image_2 = nib.load('data/raw/T2_images/RecordID_0016_T2_flair.nii').get_fdata()
T2 = np.hstack((T2, T2_image_2))
T2_image_3 = nib.load('data/raw/T2_images/RecordID_0021_T2_flair.nii').get_fdata()
T2 = np.hstack((T2, T2_image_3))
T2_image_4 = nib.load('data/raw/T2_images/RecordID_0022_T2_flair.nii').get_fdata()
T2 = np.hstack((T2, T2_image_4))
plt.hist(T2_image.flatten())
#plt.savefig('./report/istogrammi_t2')
plt.show()

with open("esempio.txt", 'w') as f:
    for elem in T2_image:
        if elem != 0:
            l.append(elem)
        k.append(l)

    f.write(str(k))
    plt.hist(k.flatten())
    # plt.savefig('./report/istogrammi_t2')
    plt.show()






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

max_1= T2_image.max()
print(max_1)

min_pixel = np.amin(T2_image)
print(min_pixel)




#histrogram
T2_2 = nib.load('data/raw/T2_images/RecordID_0012_T2_flair.nii').get_fdata()

#T2 = np.vstack(T2_image, T2_2)
plt.hist(T2_2.flatten())
#plt.savefig('./report/istogrammi_t2')
plt.show()



#create a mask
layer=33
mask = (T2_image[:,:,layer]) != 0
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
print(np.shape(T2_image_cropped))
dim1 = np.zeros((5,np.shape(T2_image_cropped)[1]))
dim2 = np.zeros((np.shape(T2_image_cropped)[0],5))

T2_image_crop=np.vstack((dim1, T2_image_cropped))
T2_image_crop=np.vstack((T2_image_crop, dim1))
T2_image_crop=np.hstack((dim2, T2_image_crop))
T2_image_crop=np.hstack((T2_image_crop, dim2))
print(np.shape(T2_image_crop))
mask=np.transpose(mask)

plt.subplot(1,4,1)
plt.imshow(T2_image[:,:,layer])
plt.title("Immagine iniziale")
plt.subplot(1,4,2)
plt.imshow(mask)
plt.title("mask")
plt.subplot(1,4,3)
plt.imshow(T2_image_cropped)
plt.title("immagine tagliata")
plt.subplot(1,4,4)
plt.imshow(T2_image_crop)
plt.title("immagine +5")
plt.style.use('grayscale')

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
max_x = 0
index_x = 0
max_y = 0
index_y = 0
max_z = 0
index_z = 0
#dict_mask = {}
o = []

T2_total=[]
for images in os.listdir(data_path):
    dict_mask = {}
    pixel ={}


    if p<=3:
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

        max_x = np.amax(i_list)
        max_y = np.amax(j_list)
        max_z = np.amax(z_list)

        min_x = np.amin(i_list)
        min_y = np.amin(j_list)
        min_z = np.amin(z_list)

        T2_image_cropped = T2_image[min_x:max_x, min_x:max_x, 0:46]

        T2_image_final = 0

        T2_total.append(T2_image_cropped[T2_image_cropped!=0])

        #print("vol in pixel: ", vol)
        dict_mask = {"num image": p,
                     "image": images,
                     "x": [np.amin(i_list), np.amax(i_list)],
                     "y": [np.amin(j_list), np.amax(j_list)],
                     "z": [np.amin(z_list), np.amax(z_list)],
                     "delta_x": (np.amax(i_list) - np.amin(i_list)),
                     "delta_y": (np.amax(j_list) - np.amin(j_list)),
                     "dim image begin" : np.shape(T2_image),
                     "dim image cropped": np.shape(T2_image_cropped),
                     "dim image +5 pixel x and y": T2_image_final,
                     "imag": T2_image}

        d.append(dict_mask)

        pixel = {"image": images,
                 "value pixel max": max_pixel,
                 "value pixel min": min_pixel}
        o.append(pixel)

        p = p + 1
        print(p)

j=[]
for elem in T2_total:
    for i in elem:
        j.append(i)
plt.hist(j)
plt.show()

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





# create file
outFileName="./pre_processing_T2_images/mask_dimension_T2_images.txt"
f=open(outFileName, "w")
for elem in d:
    f.write(str(elem))
    f.write('\n')
f.write('\n')
# find max value of images volum
f.write("----------------------------------------DIM MASK MAX VOLUM IMAGE-----------------------------------------------------")
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
    f.write(str(x_min_im-5))
    f.write(" x_max_new: ")
    f.write(str(x_max_im+5))
    f.write('\n')
    f.write("y_min_new: ")
    f.write(str(y_min_im-5))
    f.write(" y_max_new: ")
    f.write(str(y_max_im+5))
    f.write('\n')
    f.write("----------------------------------------------------------------------------------------------------------------------")
f.close()



#save pixel immagine
outFile="./pre_processing_T2_images/Pixel.txt"
g=open(outFile, "w")
for elem in o:
    g.write(str(elem))
    g.write('\n')
g.write('\n')
g.close()

#save histogram
plt.hist(j)
#plt.show()
plt.savefig('./pre_processing_T2_images/instogram_t2_final')


with open("Hystogram_T2_images.txt", 'w') as f:
    for elem in o:
        f.write(str(elem))
        f.write('\n')
    f.write('\n')
    plt.hist(j)
    plt.show()
    plt.savefig('./pre_processing_T2_images/instogram_t2_final')











with open("mask_dimension_T2_images.txt", 'w') as f:
    for elem in d:
        f.write(str(elem))
        f.write('\n')
    f.write('\n')

    # find max value of images volum
    f.write("----------------------------------------DIM MASK MAX VOLUM IMAGE-----------------------------------------------------")
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
        f.write(str(x_min_im-5))
        f.write(" x_max_new: ")
        f.write(str(x_max_im+5))
        f.write('\n')
        f.write("y_min_new: ")
        f.write(str(y_min_im-5))
        f.write(" y_max_new: ")
        f.write(str(y_max_im+5))
        f.write('\n')
        f.write("----------------------------------------------------------------------------------------------------------------------")



with open("Hystogram_T2_images.txt", 'w') as f:
    for elem in o:
        f.write(str(elem))
        f.write('\n')
    f.write('\n')
    plt.hist(j)
    plt.show()
    plt.savefig('./reports/pre_processing_T2/instogram_t2_final')




outFileName="reports/instogram_t2/nome.txt"
outFile=open(outFileName, "w")
outFile.write("""Hello my name is ABCD""")
outFile.close()


#-----------------------------------------------------------------------------------------------











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





#zero padding


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
a = np.arange(6)
a = a.reshape((2, 3))
np.pad(a, 2, pad_with)



