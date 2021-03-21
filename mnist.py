import numpy as np
import matplotlib.pyplot as plt


def load_mnist(prefix, folder):
    int_type = np.dtype('int32').newbyteorder('>')
    n_meta_data_bytes = 4 * int_type.itemsize

    data = np.fromfile(folder + "/" + prefix + '-images.idx3-ubyte', dtype='ubyte')
    magic_bytes, n_images, width, height = np.frombuffer(data[:n_meta_data_bytes].tobytes(), int_type)
    data = data[n_meta_data_bytes:].astype(dtype='float32').reshape([n_images, width, height])

    labels = np.fromfile(folder + "/" + prefix + '-labels.idx1-ubyte', dtype='ubyte')[2 * int_type.itemsize:]

    return data, labels


# def get_info(data):
#     try:
#         nb_can = data.shape[3]
#     except IndexError:
#         nb_can = 1
#     nb_elem, size, _ = data.shape
#     e_type = data.dtype
#     return nb_elem, size, nb_can, e_type
#
#
# def get_all_info(training_images, test_images):
#     nb_train, size, nb_can, p_type = get_info(training_images)
#     nb_test, _, _, l_type = get_info(test_images)
#     print('nb_train :', nb_train)
#     print('nb_test :', nb_test)
#     print('size :', size)
#     print('nb_canals :', nb_can)
#     print('pixel\'s type :', p_type)
#     print('label\'s type :', l_type)
#     return nb_train, nb_test, size, p_type, l_type


def print_info(to_print, data, labels):
    print(to_print)
    print('    data.shape:   ', data.shape)
    print('    labels.shape: ', labels.shape)
    print('    pixel\'s type:', data.dtype)
    print('    pixel\'s max: ', np.amax(data))
    print('    label\'s type:', labels.dtype)


def show_images(data, labels, rows, cols, start):
    plt.figure()
    for i in range(rows):
        for j in range(cols):
            sp = plt.subplot(rows, cols, j * rows + i + 1)
            num = start + i * rows + j
            plt.title(labels[num])
            plt.imshow(data[num])
            plt.gray()
            sp.get_xaxis().set_visible(False)
            sp.get_yaxis().set_visible(False)
    plt.show()


def preprocessing(data, labels):

    # Shuffle
    nb_data = data.shape[0]
    perm = np.random.permutation(nb_data)
    data = data[perm]
    labels = labels[perm]

    # Normalization
    data = data / 255

    return data, labels


def main():
    # Get the data and their information
    training_images, training_labels = load_mnist("train", "./data")
    test_images, test_labels = load_mnist("t10k", "./data")
    print('Before preprocessing:')
    print_info('Train set:', training_images, training_labels)
    print_info('Test set:', test_images, test_labels)
    show_images(training_images, training_labels, 3, 10, 0)

    # Preprocessing on the data
    x_train, y_train = preprocessing(training_images, training_labels)
    x_test, y_test = preprocessing(test_images, test_labels)
    print('After preprocessing:')
    print_info('Train set:', x_train, y_train)
    print_info('Test set:', x_test, y_test)
    show_images(x_train, y_train, 3, 10, 0)

    # todo git
    # todo Generator in tf


if __name__ == '__main__':
    main()

