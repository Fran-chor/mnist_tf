from keras import models
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


def basic_cnn(input_dim):
    model = models.Sequential()
    model.add(Conv2D(16, 3, activation="relu", input_shape=input_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    model.add(Conv2D(32, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    return model


def main():
    input_dim = (28, 28, 1)
    model = basic_cnn(input_dim)
    model.summary()


if __name__ == "__main__":
    main()
