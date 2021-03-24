from keras import models
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


def basic_cnn(input_dim=(28, 28, 1)):

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


# 4 ways to save a model
def save_trained_model(model):
    # mod.save("./saved_models/trained_model")
    # mod.save("./saved_models/trained_model", include_optimizer=False)
    model.save_weights("./saved_models/trained_model/")
    # tf.saved_model.save(mod, './saved_models/trained_model')


# 4 ways to load a model
def load_trained_model(model):
    # model = load_model("./saved_models/trained_model/")
    # model = load_model("./saved_models/trained_model/", compile=False)
    model.load_weights("./saved_models/trained_model/")
    # model = tf.saved_model.load("./saved_models/trained_model")


def main():
    model = basic_cnn()
    model.summary()


if __name__ == "__main__":
    main()
