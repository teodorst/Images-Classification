import cPickle
import os
import numpy as np
from PIL import Image
import sys


BATCH_NO = 5

FIRST_TRAIN_IMAGES_FILE = 'first_set_train_images.npy'
FIRST_TRAIN_CLASSES_FILE = 'first_set_train_classes.npy'
FIRST_TEST_IMAGES_FILE = 'first_set_test_images.npy'
FIRST_TEST_CLASSES_FILE = 'first_set_test_classes.npy'

SECOND_TRAIN_IMAGES_FILE = 'second_set_train_images.npy'
SECOND_TRAIN_CLASSES_FILE = 'second_set_train_classes.npy'
SECOND_TEST_IMAGES_FILE = 'second_set_test_images.npy'
SECOND_TEST_CLASSES_FILE = 'second_set_test_classes.npy'

SECOND_DATASET_PATH = 'English/Img'
SECOND_DATASET_SUBFOLDERS = ['GoodImg/Bmp', 'BadImag/Bmp']
SECOND_DATASET_NUMBER_OF_CLASSES = 10
FIRST_SET_FOLDER = 'cifar-10-batches-py'
SECOND_DATASET_FILE = 'second_dataset'
METADATES_FILE = 'batches.meta'
RED_POS = 0
GREEN_POS = 1024
BLUE_POS = 2048
IMAGE_SIZE = 32, 32


def unpickle(file):
    with open(file, 'rb') as image_file:
        image_dict = cPickle.load(image_file)
    return image_dict


def import_image(index, images, tags):
    tag_index = images['labels'][index]
    image = images['data'][index].reshape((3, 32, 32))

    tag = tags[tag_index]
    one_of_k = np.zeros((10, 1))
    one_of_k[tag_index] = 1
    image = np.array(image, dtype='float64')
    return image, tag, one_of_k

def import_first_dataset():
    train_images = []
    train_one_of_ks = []
    test_images = []
    test_one_of_ks = []
    dataset = {}

    metadates = unpickle(os.path.join(FIRST_SET_FOLDER, METADATES_FILE))
    dataset['classes'] = metadates['label_names']
    dataset['items_on_class'] = metadates['num_cases_per_batch']
    if (not os.path.exists(FIRST_TRAIN_IMAGES_FILE)) or (not os.path.exists(FIRST_TRAIN_CLASSES_FILE)) or (not os.path.exists(FIRST_TEST_IMAGES_FILE)) or (not os.path.exists(FIRST_TEST_CLASSES_FILE)):

        for batch_no in xrange(1, BATCH_NO+1):

            dates = unpickle(os.path.join(FIRST_SET_FOLDER, 'data_batch_' + str(batch_no)))
            dates_len = len(dates['labels'])

            for i in xrange(dates_len):
                img, tag, one_of_k = import_image(i, dates, dataset['classes'])
                # print "Read image %d from batch %d" % (i, batch_no)
                train_images.append(img)
                train_one_of_ks.append(one_of_k)
                # train_tags.append(tag)

        dates = unpickle(os.path.join(FIRST_SET_FOLDER, 'test_batch'))
        dates_len = len(dates['labels'])

        for i in xrange(dates_len):
            img, tag, one_of_k = import_image(i, dates, dataset['classes'])
            test_images.append(img)
            test_one_of_ks.append(one_of_k)
            # test_tags.append(tag)

        train_images = np.array(train_images)
        train_one_of_ks = np.array(train_one_of_ks)
        test_images = np.array(test_images)
        test_one_of_ks = np.array(test_one_of_ks)

        save_data(train_images, train_one_of_ks, test_images, test_one_of_ks, 1)

    else:
        train_images, train_one_of_ks, test_images, test_one_of_ks = load_data(1)

    images_standardization(train_images, test_images)

    dataset['train_images'] = train_images
    # dataset['train_tags'] = train_tags
    dataset['train_one_of_ks'] = train_one_of_ks
    dataset['test_images'] = test_images
    # dataset['test_tags'] = test_tags
    dataset['test_one_of_ks'] = test_one_of_ks
    return dataset


def import_second_dataset():
    train_images = []
    train_tags = []
    train_one_of_ks = []
    test_images = []
    test_one_of_ks = []
    test_tags = []
    dataset = {}

    if (not os.path.exists(SECOND_TRAIN_IMAGES_FILE)) or (not os.path.exists(SECOND_TRAIN_CLASSES_FILE)) or (not os.path.exists(SECOND_TEST_IMAGES_FILE)) or (not os.path.exists(SECOND_TEST_CLASSES_FILE)):

        for subfolder in SECOND_DATASET_SUBFOLDERS:
            folders_path = os.path.join(SECOND_DATASET_PATH, subfolder)
            folders = [f for f in os.listdir(folders_path) if
                       'Sample' in f][:SECOND_DATASET_NUMBER_OF_CLASSES]
            # print folders
            for class_index, class_folder in enumerate(folders):
                class_path = os.path.join(folders_path, class_folder)
                files = [f for f in os.listdir(class_path)]

                train_images_no = int(len(files) * 0.7)

                for image_index, image_file in enumerate(files):
                    # print image_file
                    image = load_image(os.path.join(class_path, image_file))
                    image_array = image_resize(image)
                    one_of_ks = np.zeros(SECOND_DATASET_NUMBER_OF_CLASSES)
                    one_of_ks[class_index] = 1
                    if image_index < train_images_no:
                        train_images.append(image_array)
                        train_one_of_ks.append(one_of_ks)
                    else:
                        test_images.append(image_array)
                        test_one_of_ks.append(one_of_ks)

        train_images = np.array(test_images)
        train_one_of_ks = np.array(test_images)
        test_images = np.array(test_images)
        test_one_of_ks = np.array(test_images)

        # Write to files
        save_data(train_images, train_one_of_ks, test_images, test_one_of_ks, 2)
    else:
        train_images, train_one_of_ks, test_images, test_one_of_ks = load_data(2)

    images_standardization(train_images, test_images)

    dataset['train_images'] = train_images
    dataset['train_one_of_ks'] = train_one_of_ks
    dataset['test_images'] = test_images
    dataset['test_one_of_ks'] = test_one_of_ks

    return dataset

def image_resize(image):
    resized_img = image.resize(IMAGE_SIZE)
    image_array = np.array(resized_img, dtype='float64')
    return image_array


def load_data(dataset_no=1):
    if dataset_no == 1:
        train_images = np.load(file=FIRST_TRAIN_IMAGES_FILE)
        train_one_of_ks = np.load(file=FIRST_TRAIN_CLASSES_FILE)
        test_images = np.load(file=FIRST_TEST_IMAGES_FILE)
        test_one_of_ks = np.load(file=FIRST_TEST_CLASSES_FILE)
    elif dataset_no == 2:
        train_images = np.load(file=SECOND_TRAIN_IMAGES_FILE)
        train_one_of_ks = np.load(file=SECOND_TRAIN_CLASSES_FILE)
        test_images = np.load(file=SECOND_TEST_IMAGES_FILE)
        test_one_of_ks = np.load(file=SECOND_TEST_CLASSES_FILE)
    else:
        print "Wrong dataset number"
        sys.exit(1)

    return train_images, train_one_of_ks, test_images, test_one_of_ks

def save_data(train_images, train_one_of_ks, test_images, test_one_of_ks, dataset_no=1):
    if dataset_no == 1:
        np.save(file=FIRST_TRAIN_IMAGES_FILE, arr=train_images)
        np.save(file=FIRST_TRAIN_CLASSES_FILE, arr=train_one_of_ks)
        np.save(file=FIRST_TEST_IMAGES_FILE, arr=test_images)
        np.save(file=FIRST_TEST_CLASSES_FILE, arr=test_one_of_ks)
    elif dataset_no == 2:
        np.save(file=SECOND_TRAIN_IMAGES_FILE, arr=train_images)
        np.save(file=SECOND_TRAIN_CLASSES_FILE, arr=train_one_of_ks)
        np.save(file=SECOND_TEST_IMAGES_FILE, arr=test_images)
        np.save(file=SECOND_TEST_CLASSES_FILE, arr=test_one_of_ks)
    else:
        print "Wrong dataset number"
        sys.exit(1)


def images_standardization(train_images, test_images):
    avg = np.mean(train_images)
    dev = np.std(train_images)
    print avg, dev
    train_images -= avg
    train_images /= dev
    test_images -= avg
    test_images /= dev


def display_image(image):
    img_display = Image.fromarray(image)
    img_display.show()

def load_image(image_path):
    return Image.open(image_path)

if __name__ == '__main__':
    import_first_dataset()
    # import_second_dataset()
