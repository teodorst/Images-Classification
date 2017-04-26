import cPickle
from os import path
import numpy as np
from PIL import Image


BATCH_NO = 5

FIRST_SET_FOLDER = 'cifar-10-batches-py'
METADATES_FILE = 'batches.meta'
RED_POS = 0
GREEN_POS = 1024
BLUE_POS = 2048

def unpickle(file):
    with open(file, 'rb') as image_file:
        image_dict = cPickle.load(image_file)
    return image_dict


def import_image(index, images, tags):
    tag_index = images['labels'][index]
    image_pixels = images['data'][index]
    image = np.empty((32, 32, 3), dtype='uint8')
    for row in xrange(32):
        for col in xrange(32):
            images_pixels_pos = row * 32 + col
            image[row][col][0] = image_pixels[images_pixels_pos+RED_POS]
            image[row][col][1] = image_pixels[images_pixels_pos+GREEN_POS]
            image[row][col][2] = image_pixels[images_pixels_pos+BLUE_POS]

    tag = tags[tag_index]
    one_of_k = np.empty((10, 1), dtype='int8')
    one_of_k[tag_index] = 1

    return image, tag, one_of_k


def import_first_set():
    train_images = []
    train_tags = []
    train_one_of_ks = []
    test_images = []
    test_one_of_ks = []
    test_tags = []
    dataset = {}

    metadates = unpickle(path.join(FIRST_SET_FOLDER, METADATES_FILE))
    dataset['classes'] = metadates['label_names']
    dataset['items_on_class'] = metadates['num_cases_per_batch']

    for batch_no in xrange(1, BATCH_NO+1):

        dates = unpickle(path.join(FIRST_SET_FOLDER, 'data_batch_' + str(batch_no)))
        dates_len = len(dates['labels'])

        for i in xrange(dates_len):
            img, tag, one_of_k = import_image(i, dates, dataset['classes'])
            print "Read image %d from batch %d" % (batch_no, i)
            train_images.append(img)
            train_one_of_ks.append(one_of_k)
            train_tags.append(tag)

        dates = unpickle(path.join(FIRST_SET_FOLDER, 'test_batch'))
        dates_len = len(dates['labels'])

        for i in xrange(dates_len):
            img, tag, one_of_k = import_image(i, dates, dataset['classes'])

            test_images.append(img)
            test_one_of_ks.append(one_of_k)
            test_tags.append(tag)

    images_standardization(train_images, test_images)

    dataset['train_images'] = train_images
    dataset['train_tags'] = train_tags
    dataset['train_one_of_ks'] = train_one_of_ks
    dataset['test_images'] = test_images
    dataset['test_tags'] = test_tags
    dataset['test_one_of_ks'] = test_one_of_ks
    return dataset

def images_standardization(train_images, test_images):
    avg = np.mean(train_images)
    dev = np.std(train_images)

    train_imgs -= avg
    train_imgs /= dev
    test_imgs -= avg
    test_imgs /= dev


def display_image(dataset):
    img_display = Image.fromarray(img, 'RGB')
    img_display.show()


if __name__ == '__main__':
    import_first_set()

