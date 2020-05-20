import cv2
import numpy as np
import os
from tqdm import tqdm

def process_data(img_size=98, shuffle=True):
    """
    A function that processes all the images in a proper way:
        grayscaled resized interpolated images
    Returns train/test/val numpy arrays;

     Attributes
    +==========+
    img_size(int) - specify it to resize image | new_dimension - (img_size, img_size) | default - 100;
    shuffle(bool) - True to shuffle the data | default - True;
    """
    data_dict = {'NORMAL': 0, 'PNEUMONIA': 1}
    data_count_dict = {'NORMAL': 0, 'PNEUMONIA': 0}
    train_data = []
    test_data = []
    val_data = []

    # Cycling through train/test/val directories
    for directory in ['train', 'test', 'val']:
        print('[+]Loading data from', directory, '...')
        # Each one has two categories: 'NORMAL' and 'PNEUMONIA'
        for category in os.listdir(directory):
            full_path = os.path.join(directory, category)
            print('\t[+]Loading data from', full_path, '...')
            # tqdm for a fancy status bar
            for f in tqdm(os.listdir(full_path)):
                try:
                    # Reading images with imread function in grayscale (more effective for the CNN)
                    img_orig = cv2.imread(os.path.join(full_path, f), cv2.IMREAD_GRAYSCALE)
                    # Resizing each one with resize function using specified size (img_size variable)
                    # Also using bicubic interpolation so that images could look fancier
                    img = cv2.resize(img_orig, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

                    # Appending train/test/val lists with [<image>, <one-hot encoded vector>]
                    if directory == 'train':
                        train_data.append([np.array(img), np.eye(2)[data_dict[category]]])
                    elif directory == 'test':
                        test_data.append(np.array([img, np.eye(2)[data_dict[category]]]))
                    else:
                        val_data.append(np.array([img, np.eye(2)[data_dict[category]]]))

                    data_count_dict[category] += 1
                except:
                    pass

    # Shuffle the data if needed
    if shuffle:
        np.random.shuffle(train_data), np.random.shuffle(test_data), np.random.shuffle(val_data)

    print('\nTotal number of images:', data_count_dict['PNEUMONIA'] + data_count_dict['NORMAL'])
    print('NORMAL:', data_count_dict['NORMAL'])
    print('PNEUMONIA:', data_count_dict['PNEUMONIA'])

    return np.array(train_data), np.array(test_data), np.array(val_data)
