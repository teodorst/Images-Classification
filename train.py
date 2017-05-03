import numpy as np                                  # Needed to work with arrays
import random
from argparse import ArgumentParser

from data_importer import import_first_dataset, import_second_dataset
from liniarize_layer import LinearizeLayer, LinearizeLayerReverse
from tanh_layer import Tanh
from softmax_layer import SoftMax
from feed_forward import FeedForward
from fullyconnected_layer import FullyConnected
from convolution_layer import ConvolutionalLayer
from relu_layer import ReluLayer
from maxpooling_layer import MaxPoolingLayer

from transfer_functions import identity, logistic, relu


def eval_nn(nn, imgs, labels, maximum = 0):
    # Compute the confusion matrix
    confusion_matrix = np.zeros((10, 10))
    correct_no = 0
    how_many = imgs.shape[0] if maximum == 0 else maximum
    for i in range(imgs.shape[0])[:how_many]:
        y = np.argmax(nn.forward(imgs[i]))
        t = np.argmax(labels[i])
        if y == t:
            correct_no += 1
        confusion_matrix[y][t] += 1

    return float(correct_no) / float(how_many), confusion_matrix / float(how_many)

def train_nn(nn, dataset, args):
    contor = 0
    train_len = len(dataset['train_images'])

    momentum_step = float((0.99 - args.momentum)) / train_len

    print 'Train len', train_len
    indeces = [i for i in xrange(train_len)]
    for i in xrange(args.total_train):
        img_index = random.choice(indeces)
        indeces.remove(img_index)
        contor += 1
        inputs = dataset["train_images"][img_index]
        targets = dataset["train_one_of_ks"][img_index]

        outputs = nn.forward(inputs)

        targets = np.reshape(targets, (10, 1))

        errors = outputs - targets

        nn.backward(inputs, errors)
        nn.update_parameters(args.learning_rate, args.momentum + i * momentum_step)

        # Evaluate the network
        if contor % args.eval_every == 0:
            train_acc, train_cm = eval_nn(nn, dataset["train_images"], dataset["train_one_of_ks"], 1000)
            test_acc, test_cm = eval_nn(nn, dataset["test_images"], dataset["test_one_of_ks"])
            nn.zero_gradients()
            print("Train acc: %2.6f ; Test acc: %2.6f" % (train_acc, test_acc))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", type = float, default = 0.001,
                        help="Learning rate")

    parser.add_argument("--momentum", type = float, default = 0.9,
                        help="Momentum")

    parser.add_argument("--eval_every", type = int, default = 200,
                        help="Learning rate")

    parser.add_argument("--total_train", type = int, default = 50000)

    parser.add_argument("--dataset", type = int, default = 1,
                        help="Dataset")
    args = parser.parse_args()

    input_size = (32, 32, 3)

    dataset = import_first_dataset() if args.dataset == 1 else import_second_dataset()
    # nn = FeedForward([LinearizeLayer(32, 32, 3), FullyConnected(32*32*3, 300, identity), Tanh(), FullyConnected(300, 10, identity), SoftMax()])

    # nn = FeedForward([LinearizeLayer(32, 32, 3), FullyConnected(32*32*3, 300, identity), Tanh(), FullyConnected(300, 10, identity), SoftMax()])

    # nn = FeedForward([ConvolutionalLayer(3, 32, 32, 3, 5, 1), ReluLayer(), MaxPoolingLayer(2),
    #     ConvolutionalLayer(3, 14, 14, 4, 3, 1), ReluLayer(), MaxPoolingLayer(2),
    #     LinearizeLayer(4, 6, 6), FullyConnected(4 * 6 * 6, 10, identity)])

    nn = FeedForward([
        ConvolutionalLayer(3, 32, 32, 6, 5, 1),
        MaxPoolingLayer(2),
        ReluLayer(),
        ConvolutionalLayer(6, 14, 14, 16, 5, 1),
        MaxPoolingLayer(2),
        ReluLayer(),
        LinearizeLayer(16, 5, 5),
        FullyConnected(400, 300, relu),
        FullyConnected(300, 10, relu),
        SoftMax()])

    # nn = FeedForward([LinearizeLayer(32, 32, 3), FullyConnected(32*32*3, 10, logistic), Tanh(), SoftMax()])

    # print nn.to_string()

    train_nn(nn, dataset, args)

