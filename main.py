import numpy as numpy
from keras import Sequential, preprocessing
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import adam
import matplotlib.pyplot as plt

input_shape_mlp = 3


def normalize(x_train, x_test):
    mu = numpy.mean(x_train, axis=0)
    std = numpy.std(x_train, axis=0)
    x_train_normalized = (x_train - mu) / std
    x_test_normalized = (x_test - mu) / std
    return x_train_normalized, x_test_normalized, mu, std


def MLP():
    dataset = numpy.loadtxt("data/training_less_random_200k.data", delimiter=";", comments='#')
    validation = numpy.loadtxt("data/validation_less_random_200k.data", delimiter=";", comments='#')
    print(dataset.shape)
    print(validation.shape)

    # normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset_scaled = scaler.fit_transform(dataset)
    validation_scaled = scaler.fit_transform(validation)

    x_train = dataset[:, 0:input_shape_mlp]
    y_train = dataset[:, input_shape_mlp:]
    print(x_train.shape)
    print(y_train.shape)

    x_test = validation[:, 0:input_shape_mlp]
    y_test = validation[:, input_shape_mlp:]
    x_train_standarized, x_test_standarized, mu, std = normalize(x_train, x_test)
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = Sequential()
    model.add(Dense(4, activation='relu', input_dim=input_shape_mlp))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(4))
    model.summary()

    opt = adam(lr=0.01, decay=1e-6)

    model.compile(
        loss='mse',
        optimizer=opt,
        metrics=['mae']
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    history = model.fit(x_train_scaled,
              y_train,
              epochs=10,
              validation_data=(x_test_scaled, y_test),
              batch_size=32)

    #first normalize data!
    prediction_data = numpy.reshape(numpy.array([100, 260, -5]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)
    test = model.predict(prediction_data_scaled)
    print("test1 {}".format(test))


    prediction_data = numpy.reshape(numpy.array([20.000, 20.000, 9]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)
    test = model.predict(prediction_data_scaled)
    print("test2 {}".format(test))


    prediction_data = numpy.reshape(numpy.array([65, 0, 6]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)
    test = model.predict(prediction_data_scaled)
    print("test3 {}".format(test))


    prediction_data = numpy.reshape(numpy.array([-140, 100, -7]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)
    test = model.predict(prediction_data_scaled)
    print("test4 {}".format(test))



    model.save('model/mlp_model1.h5')  # creates a HDF5 file

    # Plot training & validation accuracy values
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Mean absolute error of model')
    plt.ylabel('mae')
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


def MLP_rotdot():
    input_shape_mlp = 2
    dataset = numpy.loadtxt("data/training_less_random_rotdot_10k.data", delimiter=";", comments='#')
    validation = numpy.loadtxt("data/validation_less_random_rotdot_10k.data", delimiter=";", comments='#')
    print(dataset.shape)
    print(validation.shape)

    # normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))

    x_train = dataset[:, 0:input_shape_mlp]
    y_train = dataset[:, input_shape_mlp:]
    print(x_train.shape)
    print(y_train.shape)

    x_test = validation[:, 0:input_shape_mlp]
    y_test = validation[:, input_shape_mlp:]
    x_train_standarized, x_test_standarized, mu, std = normalize(x_train, x_test)
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = Sequential()
    model.add(Dense(4, activation='relu', input_dim=input_shape_mlp))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1))
    model.summary()

    opt = adam(lr=0.001, decay=1e-6)

    model.compile(
        loss='mse',
        optimizer=opt,
        metrics=['mae']
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    history = model.fit(x_train_scaled,
              y_train,
              epochs=1000,
              validation_data=(x_test_scaled, y_test),
              batch_size=32)

    #first normalize data!
    prediction_data = numpy.reshape(numpy.array([85, 120]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)
    test = model.predict(prediction_data_scaled)
    print("test1 {}".format(test))

    prediction_data = numpy.reshape(numpy.array([20.000, 20.000]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)

    test = model.predict(prediction_data_scaled)
    print("test2 {}".format(test))


    prediction_data = numpy.reshape(numpy.array([40, -9]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)
    test = model.predict(prediction_data_scaled)
    print("test3 {}".format(test))


    prediction_data = numpy.reshape(numpy.array([-140, 100]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)
    test = model.predict(prediction_data_scaled)
    print("test4 {}".format(test))
    model.save('model/mlp_model_rotdot1.h5')  # creates a HDF5 file

    # Plot training & validation accuracy values
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Mean absolute error of model')
    plt.ylabel('mae')
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

def MLP_rotdot_real():
    input_shape_mlp = 3
    dataset = numpy.loadtxt("data/realdata/VL20190322.txt.rotdot.prepared", delimiter=";", comments='#')
    validation = numpy.loadtxt("data/realdata/Vlieland.nmea.rotdot.prepared", delimiter=";", comments='#')
    print(dataset.shape)
    print(validation.shape)

    # normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))

    x_train = dataset[:, 0:input_shape_mlp]
    y_train = dataset[:, input_shape_mlp:]
    print(x_train.shape)
    print(y_train.shape)

    x_test = validation[:, 0:input_shape_mlp]
    y_test = validation[:, input_shape_mlp:]
    x_train_standarized, x_test_standarized, mu, std = normalize(x_train, x_test)
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = Sequential()
    model.add(Dense(4, activation='relu', input_dim=input_shape_mlp))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1))
    model.summary()

    opt = adam(lr=0.001, decay=1e-6)

    model.compile(
        loss='mse',
        optimizer=opt,
        metrics=['mae']
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    history = model.fit(x_train_scaled,
              y_train,
              epochs=1000,
              validation_data=(x_test_scaled, y_test),
              batch_size=32)

    #first normalize data!
    prediction_data = numpy.reshape(numpy.array([85, 120, 1]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)
    test = model.predict(prediction_data_scaled)
    print("test1 {}".format(test))

    prediction_data = numpy.reshape(numpy.array([20.000, 20.000, 1]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)

    test = model.predict(prediction_data_scaled)
    print("test2 {}".format(test))


    prediction_data = numpy.reshape(numpy.array([40, -9, 1]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)
    test = model.predict(prediction_data_scaled)
    print("test3 {}".format(test))


    prediction_data = numpy.reshape(numpy.array([-140, 100, 1]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)
    test = model.predict(prediction_data_scaled)
    print("test4 {}".format(test))

    prediction_data = numpy.reshape(numpy.array([220, 100, 1]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)
    test = model.predict(prediction_data_scaled)
    print("test5 {}".format(test))

    prediction_data = numpy.reshape(numpy.array([40, 351, 1]), [1, input_shape_mlp])
    prediction_data_normalized = (prediction_data - mu) / std
    prediction_data_scaled = scaler.transform(prediction_data)
    test = model.predict(prediction_data_scaled)
    print("test6 {}".format(test))

    model.save('model/mlp_model_real_rotdot2.h5')  # creates a HDF5 file

    # Plot training & validation accuracy values
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Mean absolute error of model')
    plt.ylabel('mae')
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


if __name__ == '__main__':
    #makes the results reproducible
    numpy.random.seed(7)
    MLP_rotdot_real()




