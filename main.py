import numpy as numpy
from keras import Sequential, preprocessing
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import adam
import matplotlib.pyplot as plt

def parse_csv(filename):
    f = open('data/' + filename)
    for line in f.readlines():
        if len(line) > 0 and line[0] == '#':
            continue
        line = line.strip().split(';')
        print(line[0])

def MLP():
    dataset = numpy.loadtxt("data/training.data", delimiter=";", comments='#')
    validation = numpy.loadtxt("data/validation.data", delimiter=";", comments='#')
    print(dataset.shape)
    print(validation.shape)

    # normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)
    validation_scaled = scaler.fit_transform(validation)

    x_train = dataset_scaled[:, 0:6]
    y_train = dataset_scaled[:, 6:]
    print(x_train.shape)
    print(y_train.shape)

    x_test = validation_scaled[:, 0:6]
    y_test = validation_scaled[:, 6:]



    model = Sequential()
    model.add(Dense(18, activation='relu', input_dim=6))
    model.add(Dense(18, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.summary()
    opt = adam(lr=0.001, decay=1e-6)

    model.compile(
        loss='mse',
        optimizer=opt,
        metrics=['accuracy']
    )

    history = model.fit(x_train,
              y_train,
              epochs=10,
              validation_data=(x_test, y_test),
              batch_size=32)

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def LSMT():
    dataset = numpy.loadtxt("data/training.data", delimiter=";", comments='#')
    validation = numpy.loadtxt("data/validation.data", delimiter=";", comments='#')
    print(dataset.shape)
    print(validation.shape)

    # normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)
    validation_scaled = scaler.fit_transform(validation)

    x_train = dataset_scaled[:, 0:6]
    y_train = dataset_scaled[:, 6:]
    print(x_train.shape)
    print(y_train.shape)

    x_train = x_train.reshape(1, 1000, 6)
    y_train = y_train.reshape(1, 1000, 3)

    x_test = validation_scaled[:, 0:6]
    y_test = validation_scaled[:, 6:]

    x_test = x_test.reshape(1, 1000, 6)
    y_test = y_test.reshape(1, 1000, 3)

    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(x_train.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(3, activation='relu', return_sequences=True))
    model.summary()
    # model.add(Dropout(0.2))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(3, activation='relu'))

    opt = adam(lr=0.001, decay=1e-6)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    model.fit(x_train,
              y_train,
              epochs=3,
              validation_data=(x_test, y_test))


if __name__ == '__main__':
    #makes the results reproducible
    numpy.random.seed(7)
    MLP()




