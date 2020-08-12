from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPool2D, Conv2D


def create_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    model = keras.Sequential([
        Conv2D(activation='relu',
               input_shape=(28, 28, 1),
               filters=32,
               kernel_size=(5, 5),
               padding='same',
               ),
        MaxPool2D(2, 2),
        Conv2D(activation='relu',
               filters=32,
               kernel_size=(5, 5),
               padding='same',
               ),
        MaxPool2D(2, 2),
        Conv2D(32, 3, activation='relu'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train.reshape(-1, 28, 28, 1), y_train_cat, epochs=5, validation_split=0.2)
    model.save('my_model.h5')
    model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test_cat)
