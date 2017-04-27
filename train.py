from data_importer import import_first_dataset, import_second_dataset


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

def train_nn(nn, dataset, args, ):
    contor = 0
    for i in np.random.permutation(dataset["train_images"])[:args['total_train']]
        contor += 1

        inputs = dataset["train_images"][i]
        label = dataset["one_of_ks"][i]
        outputs = nn.forward(inputs)
        errors = outputs - targets
        nn.backward(inputs, errors)
        nn.update_parameters(args.learning_rate)

        # Evaluate the network
        if cnt % args.eval_every == 0:
            test_acc, test_cm = eval_nn(nn, data["test_images"], data["test_classes"])
            train_acc, train_cm = eval_nn(nn, data["train_images"], data["train_classes"], 5000)
            print("Train acc: %2.6f ; Test acc: %2.6f" % (train_acc, test_acc))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", type = float, default = 0.001,
                        help="Learning rate")

    parser.add_argument("--eval_every", type = int, default = 200,
                        help="Learning rate")

    parser.add_argument("--total_train", type = int, default = 5000)

    parser.add_argument("--dataset", type = int, default = 1,
                        help="Dataset")
    args = parser.parse_args()

    input_size = (32, 32, 3)
    dataset = import_first_dataset() if args['dataset'] == 1 else import_second_dataset()
    nn = FeedForward([Liniarized(input_size), FullyConnected(300), TanH(), FullyConnected(100), SoftMax])

    train_nn(nn, args, dataset)

