import cv2
import numpy as np
import os
from tqdm import tqdm

class Augment():

    def __init__(self, image_path):
       self.image_path = image_path
       self.img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

    def add_noise(self, img):
        # Gaussian noise
        h, w = img.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (h, w))
        gauss = gauss.reshape(h, w)
        noise_img = img + gauss
        return noise_img

    def rotate(self, img):
        h, w = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D((w/2, h/2), 90, 1.0)
        img = cv2.warpAffine(img, M, (w, h))

        return img

    def blur(self, img):
        k_size = (10, 10)
        img = cv2.blur(img, k_size)
        return img

    def h_flip(self, img):
        img = img[:, ::-1]
        return img

    def v_flip(self, img):
        img = img[::-1, :]
        return img

    def augment(self, target):

        if target[0] > 0:
            img = self.img.copy()
            img_hflip = self.h_flip(img)
            img_vflip = self.v_flip(img)
            img_noise = self.add_noise(img)
            img_rot = self.rotate(img)
            img_blur = self.blur(img)

            y = np.eye(2)[0]
            return [[img_hflip, y], [img_vflip, y], [img_rot, y], [img_noise, y], [img_blur, y]]

        else:
            img = self.img.copy()
            img_noise = self.add_noise(img)
            img_rot = self.rotate(img)

            y = np.eye(2)[1]
            return [[img_rot, y], [img_noise, y]]

def process_data(img_size=98):

    """
    A function that reads images so that they are:
        grayscaled, resized, interpolated;
    Also does image augmentation: vertical/horizontal flip, rotate (45), adding noise, adjust_log;
    Returns train/test/val numpy arrays;

     Attributes
    +==========+
    img_size(int) - specify it to resize image | new_dimension - (img_size, img_size) | default - 98;
    """
    category_dict = {'NORMAL': 0, 'PNEUMONIA': 1}
    train_data = []
    test_data = []
    val_data = []

    # Cycling through train/test/val directories
    for directory in ['train', 'test', 'val']:
        print('[+] Loading data from', directory, '...')
        # Each one has two categories: 'NORMAL' and 'PNEUMONIA'
        for category in os.listdir(directory):
            full_path = os.path.join(directory, category)
            print('\t[+] Loading data from', full_path, '...')
            # tqdm for a fancy status bar
            for f in tqdm(os.listdir(full_path)):
                # Reading images in grayscale (more effective for the CNN), applying Augment class for images in train directory, then appending images as np.array to instanciated lists: train_data, test_data, val_data.

                img_gray = cv2.imread(os.path.join(full_path, f), cv2.IMREAD_GRAYSCALE)
                target = np.eye(2)[category_dict[category]]

                if directory == 'train':
                    raw_image = Augment(os.path.join(full_path, f))
                    list_of_augmented = raw_image.augment(target)
                    train_data.append([np.array(cv2.resize(img_gray, (img_size, img_size))), target])
                    for element in list_of_augmented:
                        img, target = element
                        train_data.append([np.array(cv2.resize(img, (img_size, img_size))), target])

                # Resizing each one with resize function using specified size - img_size
                img = cv2.resize(img_gray, (img_size, img_size))

                # Appending train/test/val lists with [<image>, <one-hot encoded vector>]
                if directory == 'test':
                    test_data.append(np.array([img, np.eye(2)[category_dict[category]]]))
                else:
                    val_data.append(np.array([img, np.eye(2)[category_dict[category]]]))


    # Transforming lists into np arrays, then shuffling the data
    train_data, test_data, val_data = np.array(train_data), np.array(test_data), np.array(val_data)
    np.random.shuffle(train_data), np.random.shuffle(test_data), np.random.shuffle(val_data)

    print(f'Total number of images in train:', len(train_data))
    print('\nDone.')

    return train_data, test_data, val_data