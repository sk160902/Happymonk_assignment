# Happymonk_assignment

The Given Datasets on which the experiment is to be performed are as follows:
**<u>1.MNIST dataset:** The MNIST dataset is a collection of handwritten digits that is commonly used in machine learning research. The dataset consists of 70,000 images of handwritten digits, each 28x28 pixels in size. The images are split into a training set of 60,000 images and a test set of 10,000 images. The MNIST dataset is often used as a benchmark for image classification algorithms, and has been used to train a variety of machine learning models, including deep neural networks.

**2.Wisconsin Breast Cancer dataset:** The Wisconsin Breast Cancer dataset is a collection of diagnostic measurements on breast cancer tumors. The dataset contains 569 instances, each with 30 features, including the radius, texture, perimeter, area, smoothness, and other characteristics of the tumor. The dataset is commonly used to develop and test machine learning algorithms for breast cancer diagnosis and prediction.

**3.Iris dataset:** The Iris dataset is a collection of measurements on three species of iris flowers. The dataset contains 150 instances, each with four features, including the sepal length, sepal width, petal length, and petal width of the flower. The dataset is often used as a benchmark for clustering and classification algorithms, and has been used to train a variety of machine learning models.

**4.Banknote dataset:** The Banknote dataset is a collection of images of banknotes that have been authenticated using various methods. The dataset contains 1,372 instances, each with five features, including the variance, skewness, kurtosis, and entropy of the image. The dataset is commonly used to develop and test machine learning algorithms for banknote authentication and fraud detection.
The Python script defines two functions, create_model() and train_model(), which can be used to create and train neural network models for various datasets, and then evaluate the performance of the models based on their loss and accuracy. The script uses the Keras API provided by TensorFlow to create and train the models.

The plots of the to visualize the distribution of specific features in these datasets for each species.

**Code-Description:**

The code trains an artificial neural network (ANN) model on four different datasets: Bank-Note, Iris, Wisconsin Breast Cancer, and MNIST. The ANN model is defined with one hidden layer, where the output layer has a softmax activation function, and the loss function used during training is sparse categorical cross-entropy. The model is trained multiple times with different activation functions (sigmoid, tanh, relu, and softmax) and random values for k0 and k1. The function train_model() takes as input the name of the dataset to train on, the number of times the model is trained, the number of epochs to run the training, and a flag to indicate whether or not to plot the loss vs. epochs.



The create_model() function takes as the name of a dataset and optional values for k0 and k1 as input, which are scaling and shifting parameters that are randomly generated from a normal distribution and then applied to the input data. The function normalizes the input data and returns the model along with the training and testing data. If the dataset is Bank-Note, the function loads the data from a specified URL and sets the number of classes to 2. If the dataset is Iris, the function loads the data using the specified url and sets the number of classes to 3. If the dataset is Wisconsin Breast Cancer, the function loads the data using the load_breast_cancer() function from scikit-learn and sets the number of classes to 2. If the dataset is MNIST, the function loads the data using the keras.datasets.mnist.load_data() function and sets the number of classes to 10. If the dataset name is not recognized, the function raises a ValueError. The function loads the specified dataset, preprocesses the data (reshapes and normalizes it), defines a simple Artificial neural network model with one hidden layer, and compiles the model with the Adam optimizer and sparse categorical cross-entropy loss function. The function returns the model and the preprocessed data, split into training and testing sets.

The train_model() function uses the create_model() function to load the specified dataset and create the ANN model.The function takes  the name of a dataset, the number of runs to perform, the number of epochs to train for as input, and a boolean value indicating whether to plot the loss vs. epochs during training.If the plot_loss flag is set to True, the function plots the loss vs. epochs for each run of the model training. The function iterates over a list of activation functions (sigmoid, tanh, relu, and softmax), and for each activation function, it performs the specified number of runs of the create_model() function with random values of k0 and k1, trains the model for the specified number of epochs, evaluates the performance of the model on the test set using the f1 score, and stores the accuracy and f1 score. The function then selects the activation function that gives the highest average accuracy, and returns the training and testing losses, training and testing accuracies, and the best f1 score.

The script also imports several Python modules, including TensorFlow, NumPy, Matplotlib, and scikit-learn, and uses them for various purposes. For example, scikit-learn is used to load some of the datasets, and the f1_score() function is used to compute the f1 score during evaluation. Matplotlib is used to plot the loss vs. epochs during training.


Overall, the code performs a simple hyperparameter search to find the best activation function for the ANN model.


**OUTPUTS:**

**1.Implementing the Algorithm over the MNSIT-dataset:**
**2.Implementing the Algorithm over the Wisconsin Breast Cancer-dataset:**
**3.Implementing the Algorithm over the Iris-dataset:**
**4.Implementing the Algorithm over the Bank-Note-dataset:**
