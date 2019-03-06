import numpy as numpy
from keras import Sequential, preprocessing
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import adam


def parse_csv(filename):
    f = open('data/' + filename)
    for line in f.readlines():
        if len(line) > 0 and line[0] == '#':
            continue
        line = line.strip().split(';')
        print(line[0])



if __name__ == '__main__':
    #makes the results reproducible
    numpy.random.seed(7)

    dataset = numpy.loadtxt("data/training.data", delimiter=";", comments='#')
    validation = numpy.loadtxt("data/validation.data", delimiter=";", comments='#')
    print(dataset.shape)
    print(validation.shape)


    #normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)
    validation_scaled = scaler.fit_transform(validation)

    x_train = dataset_scaled[:, 0:6]
    y_train = dataset_scaled[:, 6:]
    print(x_train.shape)
    print(y_train.shape)

    x_train = x_train.reshape(1, 1000, 6)
    #y_train = y_train.reshape(1, 1000, 3)

    x_test = validation_scaled[:, 0:6]
    y_test = validation_scaled[:, 6:]

    x_test = x_test.reshape(1, 1000, 6)
    #y_test = y_test.reshape(1, 1000, 3)

    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(x_train.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

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




