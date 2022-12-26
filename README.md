# CIFAR-100 Image Classification with a Siamese Neural Network
This code demonstrates how to train a Siamese neural network for image classification on the CIFAR-100 dataset. The CIFAR-100 dataset consists of 100 classes of tiny natural images, each class containing 500 training and 100 testing images. The classes are organized into 20 superclasses, and the images are of size 32x32 pixels with 3 color channels (RGB).

## Dependencies
This code requires the following packages:

* `numpy`
* `keras`
* `pickle`
* `scipy`

## Usage
To train the model, run the following command:

```python
python Cifar-100.py
```
The trained model and its performance on the test data will be saved to the `saved_models` directory.

## Model Architecture
The model consists of two identical convolutional neural network (CNN) branches, each of which takes an image as input and processes it through a series of convolutional and max pooling layers. The outputs of the two CNN branches are then compared using the L1 distance measure, and the resulting distance is passed through a dense layer with a sigmoid activation function to produce a prediction of whether the two images are similar or not.

During training, the model is presented with pairs of images, and the output is a binary label indicating whether the two images are of the same class or not. The model is compiled and fit using the CIFAR-100 training and test data, and the performance of the model on the test data is saved to a pickle file.

## Evaluation
The performance of the trained model on the test data is printed to the console, and it is also saved to a pickle file for later analysis.

## Testing
To test the model, run the following command:

```python
python Test_Cifar-100.py 
```
This code tests a trained CIFAR-100 model by predicting the class labels of the first 32 images in the test set and comparing the predictions to the true labels. The CIFAR-100 dataset contains 100 fine-grained classes, such as "beaver" or "dolphin". The model is loaded from a saved JSON file and H5 weights file, and the test set is loaded in fine label mode. The results of the predictions could be used for further analysis or evaluation.
