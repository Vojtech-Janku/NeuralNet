## University project - Neural Network from Scratch
As a semester project, I was tasked to implement a simple neural network (MLP - Multi-Layer Perceptron) from scratch in C++, without the use of advanced libraries or frameworks (for example <algorithm> was not allowed, so even matrix multiplication had to be implemented personally).
Now I am gradually expanding the project to contain more methods, layer types, optimizers etc.

### USER GUIDE
- the model is created, trained and used for predictions in the main function in neural_net.cpp, you can play around with hyperparameters there - I will make some more user friendly API in time
- compile and run the program by running <run.sh>
- running the program outputs two files to the data directory:
     - `train_predictions.csv` - network predictions for the train set.
     - `test_predictions.csv`  - network predictions for the test set.

- download and prepare the data by running <data/download_data.py>
- you can display the image at [index] of the training set with the corresponding label by running <data/display_data.py> [index]

### DATASET
Fashion MNIST (https://arxiv.org/pdf/1708.07747.pdf) a modern version of a
well-known MNIST (http://yann.lecun.com/exdb/mnist/). It is a dataset of
Zalando's article images ‒ consisting of a training set of 60,000 examples
and a test set of 10,000 examples. Each example is a 28x28 grayscale image,
associated with a label from 10 classes. The dataset is in CSV format. There
are four data files in the data folder, split from the original files:
 - `fashion_mnist_train_vectors.csv`   - training input vectors
 - `fashion_mnist_test_vectors.csv`    - testing input vectors
 - `fashion_mnist_train_labels.csv`    - training labels
 - `fashion_mnist_test_labels.csv`     - testing labels
