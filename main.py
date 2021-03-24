import mnist_deep.mnist as mnist
import mnist_deep.my_models as my_models
import mnist_deep.training as training
import mnist_deep.analysis as analysis
import tensorflow as tf
from tensorflow.keras.models import load_model

# TODO See the tutorial on github (in my favorites) to improve this code
# For instance, using repeat in the dataset and doing by step instead of epoch ???
# TODO See how to do a Notebooks


def main():

    # TODO put these variables in argument
    epochs = 1
    eager_mode = True
    running = "analysis"

    train_ds, test_ds = mnist.load_mnist_dataset()

    model = my_models.basic_cnn()

    if running == "training":

        optimizer, loss_fn, train_loss, train_metric, valid_loss, valid_metric = \
            training.initialization(eager_mode)

        training.train(model, train_ds, test_ds, optimizer, loss_fn, epochs,
                       train_loss, train_metric, valid_loss, valid_metric)

    elif running == "analysis":

        my_models.load_trained_model(model)

        analysis.get_accuracy(model, test_ds)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
