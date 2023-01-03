# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


class SudokuNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        input_shape = (height, width, depth)
        # premier ensemble de couches CONV => RELU => POOL
        model.add(Conv2D(32, (5, 5), padding="same",
                         input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # second ensemble de couches CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # premier ensemble de couches FC => RELU
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        # second ensemble de couches FC => RELU
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        # classificateur softmax
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # retourner l'architecture réseau construite
        return model


test = SudokuNet.build(
    28,
    28,
    1,
    10
)
test.summary()