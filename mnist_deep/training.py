import tensorflow as tf
import matplotlib.pyplot as plt
import mnist_deep.my_models as my_models
from mnist_deep.utils import tfw_print
import time
import datetime


# To make an iteration of training considering a batch
@tf.function
def train_step(mod, opt, loss_fn, x, y, train_loss, train_metric):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        pred = mod(x, training=True)  # take and return a tf.tensor
        loss = loss_fn(y, pred)
    grads = tape.gradient(loss, mod.trainable_variables)
    opt.apply_gradients(zip(grads, mod.trainable_variables))

    train_loss(loss)
    train_metric(y, pred)


# To make an iteration of testing, considering a batch
@tf.function
def valid_step(mod, loss_fn, x, y, valid_loss, valid_metric):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    pred = mod(x, training=False)  # take and return a tf.tensor
    val_loss = loss_fn(y, pred)

    valid_loss(val_loss)
    valid_metric(y, pred)


def train(mod, train_ds, valid_ds, opt, loss_fn, epo, train_loss, train_metric,
          valid_loss, valid_metric, train_summary_writer, valid_summary_writer):

    # The lists used to store the loss and metric information, to plot them after
    train_loss_results = []
    train_metric_results = []
    valid_loss_results = []
    valid_metric_results = []

    start_training = time.time()

    for e in range(epo):
        # Reset the metrics at the start of each epoch
        train_loss.reset_states()
        train_metric.reset_states()
        valid_loss.reset_states()
        valid_metric.reset_states()

        for x, y in train_ds:
            train_step(mod, opt, loss_fn, x, y, train_loss, train_metric)

            # Writing in files the information of loss and metric,
            # to be able to plot them in Tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=e)
                tf.summary.scalar('accuracy', train_metric.result(), step=e)

        for x, y in valid_ds:
            valid_step(mod, loss_fn, x, y, valid_loss, valid_metric)

            # Writing in files the information of loss and metric,
            # to be able to plot them in Tensorboard
            with valid_summary_writer.as_default():
                tf.summary.scalar('loss', valid_loss.result(), step=e)
                tf.summary.scalar('accuracy', valid_metric.result(), step=e)

        # Using print would be better, tf.print not necessary here
        tfw_print(
            "Epoch {:03d},".format(e + 1),
            "Loss: {:.4f},".format(train_loss.result()),
            "Accuracy: {:.2%},".format(train_metric.result()),
            "Valid Loss: {:.4f},".format(valid_loss.result()),
            "Valid Accuracy: {:.2%}".format(valid_metric.result())
        )

        # The lists used to store the losses and metrics information, to plot them after
        train_loss_results.append(train_loss.result())
        train_metric_results.append(train_metric.result())
        valid_loss_results.append(valid_loss.result())
        valid_metric_results.append(valid_metric.result())

    stop_training = time.time()

    # Computation of the training time
    duration = stop_training - start_training
    # Using print would be better, tf.print not necessary here
    tfw_print("Total training time: {:.1f}s".format(duration))
    tfw_print("Training time per epoch: {:.1f}s".format(duration / epo))

    # To save the weight of the model, in order to reuse it
    my_models.save_trained_model(mod)

    return train_loss_results, train_metric_results, valid_loss_results, valid_metric_results


# Creations of useful instances for the training
def initialization(eager_mode):
    # Enable or not the "eager mode"
    # Faster when using the "graphs mode"
    tf.config.run_functions_eagerly(eager_mode)

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_metric = tf.keras.metrics.CategoricalAccuracy(name='train_metric')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_metric = tf.keras.metrics.CategoricalAccuracy(name='valid_metric')

    # Writers useful to write in files the information of losses and metrics,
    # which can be used to plot them in Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    valid_log_dir = 'logs/gradient_tape/' + current_time + '/valid'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    return optimizer, loss_fn, train_loss, train_metric, valid_loss, valid_metric, \
        train_summary_writer, valid_summary_writer


# To plot the curves of losses or metrics
def plot_curves(train_list, valid_list, info_type, ymax=-1):
    if ymax == -1:
        ymax = max(max(train_list), max(valid_list))
    plt.figure()
    plt.plot(train_list, color='blue', linestyle='--', label="Train " + info_type)
    plt.plot(valid_list, color='green', label="Valid " + info_type)
    plt.xlabel("Epochs")
    plt.ylim(0, ymax)
    plt.title("Evolution of : " + info_type)
    plt.legend()
    plt.draw()
    plt.pause(0.001)


def main():
    pass


if __name__ == "__main__":
    main()
