import mnist_deep.mnist as mnist
import mnist_deep.models as models
import mnist_deep.training as training


def main():

    # TODO put these variables in argument
    epochs = 2
    eager_mode = True

    train_ds, test_ds = mnist.load_mnist_dataset()

    model = models.basic_cnn()

    optimizer, loss_fn, train_loss, train_metric, valid_loss, valid_metric = \
        training.initialization(eager_mode)

    training.train(model, train_ds, test_ds, optimizer, loss_fn, epochs,
                   train_loss, train_metric, valid_loss, valid_metric)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
