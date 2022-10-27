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

    for images in os.listdir(data_path):
        dict_mask = {}
        pixel = {}


        path_image = os.path.join(data_path, images)
        if path_image.find('DS_Store') != -1:
            continue

        T2_image = nib.load(path_image).get_fdata()
        T2_image = np.rot90(T2_image)
        nd = np.ndim(T2_image)

        max_pixel = np.amax(T2_image)
        min_pixel = np.amin(T2_image)

        mask = ((T2_image[:, :, :]) > 0)

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

        p = p + 1
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
outFileName = "./pre_processing_T2_images/mask_dimension_T2_images.txt"
f = open(outFileName, "w")
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
outFile = "./pre_processing_T2_images/Pixel.txt"
g = open(outFile, "w")
for elem in o:
    g.write(str(elem))
    g.write('\n')
g.write('\n')
g.close()


#save histogram
plt.hist(final_hist)
plt.savefig('./pre_processing_T2_images/instogram_t2_final')





