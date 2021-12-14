import tensorflow as tf
import numpy as np


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


def make_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten()) # Flatten the 28x28 to 784x1
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # Only 10 output nodes for 0-9 with a probability distribution of softmax
    return model


def train(model):
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)

    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)

    model.save("mnist.model") # saving model


def predict(model):
    pred = model.predict(np.expand_dims(x_test[0], axis=0))
    print(np.argmax(pred))


if __name__ == "__main__":
    model = make_model()
    train(model)