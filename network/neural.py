import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plot
import parse


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


def adult():
    """
    The adult/census income dataset. Binary classification, large dataset. (45222 rows, 87 columns)
    :return:
    """

    print("\n\nThe adult/census income dataset. A binary classification problem.")
    data, target = parse.get_adult_list()
    train_data, test_data, train_target, test_target = model_selection.train_test_split(data, target)
    train_data = preprocessing.scale(train_data)
    test_data = preprocessing.scale(test_data)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(train_data.shape[1],)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
    model.fit(train_data, train_target, epochs=200, validation_split=0.2, verbose=0, callbacks=[early_stop])
    accuracy = model.evaluate(test_data, test_target, verbose=0)
    print("Accuracy of adult/census income predictions: {:2.1f}%".format(accuracy[1]*100))


def cars():
    """
    The cars dataset. It has roughly 1700 instances, and is a classification problem.
    :return:
    """
    print("\n\nThe cars dataset, a classification problem.")
    data, target = parse.get_cars_list()
    train_data, test_data, train_target, test_target = model_selection.train_test_split(data, target)
    train_data = preprocessing.scale(train_data)
    test_data = preprocessing.scale(test_data)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(train_data.shape[1],)),
        keras.layers.Dense(128, activation=tf.nn.sigmoid),
        keras.layers.Dense(128, activation=tf.nn.sigmoid),
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    model.fit(train_data, train_target, epochs=500, validation_split=0.2, verbose=0, callbacks=[early_stop])
    accuracy = model.evaluate(test_data, test_target, verbose=0)
    print("The accuracy of the algorithm: {:2.1f}%".format(accuracy[1] * 100))
    print("The loss of the algorithm: {:1.4f}".format(accuracy[0]))


def mpg():
    """
    The MPG dataset. It has a small number of instances at about 400.
    This dataset has been the most difficult for me to predict with any accuracy. Probably a large part of that is
    the relatively small dataset
    :return:
    """
    print("\n\nThe MPG dataset, a regression problem.")
    data, target = parse.get_mpg_list()
    train_data, test_data, train_target, test_target = model_selection.train_test_split(data, target)
    train_data = preprocessing.scale(train_data)
    test_data = preprocessing.scale(test_data)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(), metrics=['mae'])
    model.fit(train_data, train_target, epochs=250, verbose=0)
    loss, mae = model.evaluate(test_data, test_target, verbose=0)
    test_predictions = model.predict(test_data).flatten()

    right = sum([1 for i in range(len(test_predictions)) if abs(test_predictions[i] - test_target[i]) < mae])
    print("Accuracy with delta of mae, or ±{:1.2f} MPG. Accuracy = {:2.1f}%".format(mae,
                                                                                    (right / len(test_predictions)
                                                                                     * 100)))
    print("MSE loss: {:4.2f}".format(loss))


def fashion():
    """
    The built in fashion dataset. It uses images, whatever kind of problem that is.
    It also has 60,000 rows, so I'd say it is significantly large in that respect.
    Type: Images
    Essentially followed a tutorial on https://www.tensorflow.org/tutorials/
    :return:
    """
    print("\n\nThe fashion dataset from Keras. It's a classification problem with significant rows")
    fashion_minst = keras.datasets.fashion_mnist
    (train_images, train_lables), (test_images, test_labels) = fashion_minst.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                   'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.elu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_lables, epochs=15, verbose=0)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

    print("Fashion Test Accuracy: {:2.1f}".format(test_acc * 100))
    print("Fashion Loss: {:1.3f}".format(test_loss))


if __name__ == "__main__":
    # fashion()
    mpg()
    # cars()
    # adult()
