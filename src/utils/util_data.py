import torch
import os
import random
from scipy.ndimage import shift
import nibabel as nib
import numpy as np
from scipy.stats import norm


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_img(img_path):
    img = nib.load(img_path)
    img = img.get_fdata()
    return img


def num2vect(x, bin_range, bin_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers


def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example:
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def cropping_T2(img, delta_x_max, delta_y_max):

    mask = (img[:, :, :] != 0)  # create a mask
    y = np.where(np.any(mask, axis=0))[0]  # value im !=0 y
    y_min, y_max = y[[0, -1]]              # max and min y

    x = np.where(np.any(mask, axis=1))[0]  # value im !=0 x
    x_min, x_max = x[[0, -1]]              # max and min x

    x_star = int(delta_x_max/2)                # required to have the same size as parallelepiped along x --> delta_x_max/2
    y_star = int(delta_y_max/2)                # required to have the same size as parallelepiped along y --> delta_y_max/2

    delta_x_imag = int((x_max - x_min)/2)  # find delta_x image i-esim
    delta_y_imag = int((y_max - y_min)/2)  # find delta_y image i-esim

    x_min_im = x_min + delta_x_imag - x_star - 1  # find the x min for the new image cropped
    x_max_im = x_min + delta_x_imag + x_star      # find the x max for the new image cropped

    y_min_im = y_min + delta_y_imag - y_star - 1  # find the y min for the new image cropped
    y_max_im = y_min + delta_y_imag + y_star      # find the y max for the new image cropped


    T2_image_final = img[x_min_im:x_max_im, y_min_im:y_max_im, :]  # image final
    T2_image_final = np.pad(T2_image_final, 5, pad_with)           # zero padding --> addition of null pixels to the image
    T2_image_cropped = T2_image_final[:, :, 0:47]
    #print(T2_image_cropped.shape)

    return T2_image_cropped


def augmentation(img):
    # shift
    r = random.randint(0, 100)
    if r < 70:
        r1 = random.randint(-3, 3)
        r2 = random.randint(-3, 3)
        r3 = random.randint(-3, 3)
        img = np.roll(img, (r1, r2, r3), axis=(0, 1, 2))
    # flip
    r = random.randint(0, 100)
    if r < 30:
        img = np.flip(img, 0).copy()
    return img


def loader(img_path, img_dim, clip=None, norm=None, step="train"):
    # Img
    img = load_img(img_path)
    # Clip
    if clip:
        img = np.clip(img, clip['min'], clip['max'])
    # Norm
    if norm:
        if norm == "max_scaler_01":
            img = img / clip['max']
        elif norm == "minmax_scaler_-11":
            img = (((img - clip['min']) * (1 - (-1))) / (clip['max'] - clip['min'])) + (-1)
    # Cropping
    img = crop_center(img, (img_dim['x'], img_dim['y'], img_dim['z']))
    if step == "train":
        img = augmentation(img)
    # To Tensor
    img = torch.Tensor(img)
    img = torch.unsqueeze(img, dim=0)
    return img


def loader_T2(img_path, img_dim, clip=None, norm=None, step="train"):
    # Img
    img = load_img(img_path)
    #cropping   --> dx_max/ dy_max
    img = cropping_T2(img, 325, 396)

    # Clip
    if clip:
        img = np.clip(img, clip['min'], clip['max'])
    # Norm
    if norm:
        if norm == "max_scaler_01":
            img = img / clip['max']
        elif norm == "minmax_scaler_-11":
            img = (((img - clip['min']) * (1 - (-1))) / (clip['max'] - clip['min'])) + (-1)
    # Cropping
    #img = crop_center(img, (img_dim['x'], img_dim['y'], img_dim['z']))
    if step == "train":
        img = augmentation(img)
    # To Tensor
    img = torch.Tensor(img)
    img = torch.unsqueeze(img, dim=0)
    return img


class ImgDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, classes, cfg_data, step):
        'Initialization'
        self.step = step
        self.img_dir = cfg_data["img_dir"]
        self.data = data
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        # Clip
        self.clip = cfg_data["clip"]
        # Norm
        self.norm = cfg_data["norm"]
        # Dim
        self.img_dim = cfg_data["img_dim"]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.data.iloc[index]
        id = row.name
        img_file = row.img
        y = row.label
        # Load data and get label
        img_path = os.path.join(self.img_dir, img_file)
        x = loader(img_path=img_path, img_dim=self.img_dim, clip=self.clip, norm=self.norm, step=self.step)
        return x, self.class_to_idx[y], id
