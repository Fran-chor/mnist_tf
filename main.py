import getopt
import sys
import matplotlib.pyplot as plt
import tensorflow as tf

import mnist_deep.analysis as analysis
import mnist_deep.mnist as mnist
import mnist_deep.my_models as my_models
import mnist_deep.training as training


# TODO See the tutorial on github (in my favorites) to improve this code
# TODO See how to do a Notebooks

def usage():
    print("Usage: {} -m <training/analysis> -e <epochs>".format(sys.argv[0]))
    print("-g to use the graphs mode (eager_mode by default)")
    print("-h to print the usage")


def get_args():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hm:ge:", ["help", "mode=", "graphs", "epochs="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    mode = None
    graphs = None
    epochs = None
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-m", "--mode"):
            if arg in ("training", "analysis"):
                mode = arg
            else:
                usage()
                sys.exit(2)
        elif opt in ("-g", "--graphs"):
            graphs = True
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        else:
            assert False, "unhandled option"
    return mode, graphs, epochs


def main():

    mode, graphs, epochs = get_args()
    eager_mode = not graphs

    train_ds, test_ds = mnist.load_mnist_dataset()

    model = my_models.basic_cnn()

    if mode == "training":

        optimizer, loss_fn, train_loss, train_metric, valid_loss, valid_metric = \
            training.initialization(eager_mode)

        train_loss_results, train_metric_results, valid_loss_results, valid_metric_results = \
            training.train(model, train_ds, test_ds, optimizer, loss_fn, epochs,
                           train_loss, train_metric, valid_loss, valid_metric)

        training.plot_curves(train_loss_results, valid_loss_results, "loss")
        training.plot_curves(train_metric_results, valid_metric_results, "accuracy")

        # Necessary to keep the figures open
        plt.show()

    elif mode == "analysis":

        my_models.load_trained_model(model)

        tf.config.run_functions_eagerly(eager_mode)

        analysis.get_accuracy(model, test_ds)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
