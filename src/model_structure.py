from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization


# define different model structure here.
class All_models():
    def Simple_NeuralNet(number_classes):
        model = Sequential()
        model.add(Dense(units=256, activation='relu', input_dim=128))
        model.add(BatchNormalization())
        model.add(Dense(units=1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(units=number_classes, activation='softmax'))
        return model 