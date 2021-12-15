import tensorflow as tf
import numpy as np
from itertools import permutations
import sys


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


def make_model(layers=[280, 128, 60]):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten()) # Flatten the 28x28 to 784x1
    for layer in layers:
        model.add(tf.keras.layers.Dense(layer, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # Only 10 output nodes for 0-9 with a probability distribution of softmax
    return model


def train(model, epochs):
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, verbose=0)

    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    # print(val_loss, val_acc)

    return val_loss, val_acc


def tune_hyperparameters():
    # Generates 180 different network types
    # Best found was (layers=[250, 150], epochs=5)
    number_of_epochs = [3, 5, 10]
    neurons = [50, 100, 150, 250]
    number_of_layers = [2, 3, 4]

    configs = []
    for e in number_of_epochs:
        for l in number_of_layers:
            for layers in permutations(neurons, l):
                configs.append((layers, e))
    
    best_loss = float('inf')
    for i, (layer, e) in enumerate(configs):
        print("%.2f percent complete, Best Loss: %.2f" % (round(i / len(configs) * 100, 3), best_loss), end='\r')
        sys.stdout.flush()

        model = make_model(layer)
        loss, acc = train(model, e)
        if loss < best_loss:
            best_loss = loss
            model.save("mnist.model") # saving model

            print("Best config: layers={0}, epochs={1}".format(layer, e))
            print("\tLoss: {0}".format(loss))
            print("\tAccuracy: {0}".format(acc))
    


def load_and_predict():
    model = tf.keras.models.load_model("mnist.model")
    pred = model.predict(x_test)
    predictions = [np.argmax(p) for p in pred]
    confusion = np.zeros((10, 10))
    for p, a in zip(predictions, y_test):
        confusion[p][a] += 1

    print("The confusion matrix:")
    print(confusion.astype(int))
        


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "-train":
            tune_hyperparameters()
            load_and_predict()
        else:
            raise ValueError("Unknown flag: %s" % sys.argv[1]) from None
    else:
        load_and_predict()
