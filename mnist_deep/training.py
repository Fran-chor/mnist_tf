import tensorflow as tf
import matplotlib.pyplot as plt

# TODO test and see the difference with and without eager mode (add a time criteria too)
# TODO Add the curves of the losses and metrics, make it work without eager mode


@tf.function
def train_step(mod, opt, loss_fn, x, y, train_loss, train_metric):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        pred = mod(x, training=True)
        loss = loss_fn(y, pred)
    grads = tape.gradient(loss, mod.trainable_variables)
    opt.apply_gradients(zip(grads, mod.trainable_variables))

    train_loss(loss)
    train_metric(y, pred)


@tf.function
def valid_step(mod, loss_fn, x, y, valid_loss, valid_metric):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    pred = mod(x, training=False)
    val_loss = loss_fn(y, pred)

    valid_loss(val_loss)
    valid_metric(y, pred)


@tf.function
def train(mod, train_ds, valid_ds, opt, loss_fn, epo, train_loss, train_metric, valid_loss, valid_metric):
    for e in range(epo):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_metric.reset_states()
        valid_loss.reset_states()
        valid_metric.reset_states()

        for x, y in train_ds:
            train_step(mod, opt, loss_fn, x, y, train_loss, train_metric)

        for x, y in valid_ds:
            valid_step(mod, loss_fn, x, y, valid_loss, valid_metric)

        tf.print(
            f'Epoch {e + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_metric.result() * 100}, '
            f'Valid Loss: {valid_loss.result()}, '
            f'Valid Accuracy: {valid_metric.result() * 100}'
        )


def initialization(eager_mode):
    tf.config.run_functions_eagerly(eager_mode)

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_metric = tf.keras.metrics.CategoricalAccuracy(name='train_metric')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_metric = tf.keras.metrics.CategoricalAccuracy(name='valid_metric')

    return optimizer, loss_fn, train_loss, train_metric, valid_loss, valid_metric


def plot_curves(train_list, valid_list, info_type, ymax=1):
    plt.figure()
    plt.plot(train_list, color='blue', linestyle='--', label="Train " + info_type)
    plt.plot(valid_list, color='green', label="Valid " + info_type)
    plt.xlabel("Epochs")
    plt.ylim(0, ymax)
    plt.title("Evolution of : " + info_type)
    plt.legend()


def main():
    pass


if __name__ == "__main__":
    main()
