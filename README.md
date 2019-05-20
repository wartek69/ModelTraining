# ModelTraining
This project is used to train a model that approximates a vessel using a multilayer perceptron (MLP) neural network for the master thesis of autonomous navigation of inland vessels.

# Different models
This script supports two types of models to be trained. The first one is a black box model, the second on the rotdot model. In the picture below the difference can be seen.
[[https://github.com/wartek69/ModelTraining.git/master/img/models.png|alt=models]]
Since the rotdot model is less complex, the training of it is easier.
The methods that can be used to train respectively the blackbox model and the rotdot model are MLP() and MLP_rotdot()

# Normalisation
Normalisation of the data is necessary since otherwise the different inputs would not have an equal weight of importance for the model. Two ways of normalisation were tried. The first one can be found in the normalize() method. This method uses standarization to normalise (standarise) the data. This is done by scaling the mean of the data set to 0 and the standard deviation to 1.
Another method that can be used is by scaling all the input data between 1 and -1.
In the script this is achieved by using the MinMaxScaler from the sklearn library.
Take note that it is of great importance to scale the validation set with the same scaling factor that is used in the training set! If that is not done, the training set would indirectly alter the validation set and thus breaking the concept of validation.
Normalisation of the output values is not necessary.

# Activation function
In the script the relu is used as activation function for both the models. This function is lightweight and showed to be a good match for this kind of problem. If the dying relu problem would show itself it would be interesting to migrate to the leaky relu or the swish actication function of google. Both these activation functions have a slight slope on the negative side, thus allowing the model to train there aswell.
Take note that the output layer should be a linear function since we are using regression. Our output values should be directly mapped onto the influence of our data. If we take a different activation function for the output layer, the output data of the training sets will never be achieved since the activation function would map it to different values and thus the training would be bogus.

# Size of the neural network
The objective was to keep the size of the neural network as small as possible since the performance of the predictions is a factor of great interest. The smaller the network, the more perfomance can be achieved.

# Training on real data
The objective was to eventually train a model using real data. There is a method called MLP_rotdot_real(). This method is identical to MLP_rotdot model with the only difference that it uses real data and has some debug prints.
The data that is used is not of great quality since it contains data that does not react on the input data, and thus is the result of the training a model that cannot be used.
Since no good data could be provided, this part is future work.
It is important that the provided data has the input to the vessel and the reaction of the vessel on that input. Above that it should be known whether the vessel is loaded or not loaded at the time. Only then a good model can be trained.

# Export of the model
Both the models are exported in a h5 format that can be imported pretty easily using keras. It is also important to export the normalisation factor since when you want to make predictions using your own data, that data has to be normalised using the training normalisation factor.
In this project it is chosen to export the training set and calculate the normalisation factors when initialising the model predictive control.
