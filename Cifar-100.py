"""
Module to load CIFAR-100 datasets of tiny natural images.
See http://www.cs.toronto.edu/~kriz/cifar.html

The CIFAR datasets are labeled subsets of the 80 million tiny images dataset 
collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
The images are of size 32x32 pixels with 3 color channels (RGB).

CIFAR-10
    10 classes containing 5000 training and 1000 test images, each.
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

CIFAR-100
    100 classes containing 600 images each (500 training and 100 testing).
    The classes (fine labels) are grouped into 20 superclasses (coarse labels).

Superclass                      Classes
aquatic mammals                 mammals beaver, dolphin, otter, seal, whale
fish                            aquarium fish, flatfish, ray, shark, trout
flowers                         orchids, poppies, roses, sunflowers, tulips
food containers                 bottles, bowls, cans, cups, plates
fruit and vegetables            apples, mushrooms, oranges, pears, sweet peppers
household electrical devices    clock, computer keyboard, lamp, telephone, TV
household furniture             bed, chair, couch, table, wardrobe
insects                         bee, beetle, butterfly, caterpillar, cockroach
large carnivores                bear, leopard, lion, tiger, wolf
large man-made outdoor things   bridge, castle, house, road, skyscraper
large natural outdoor scenes    cloud, forest, mountain, plain, sea
large omnivores and herbivores  camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals            fox, porcupine, possum, raccoon, skunk
non-insect invertebrates        crab, lobster, snail, spider, worm
people                          baby, boy, girl, man, woman
reptiles                        crocodile, dinosaur, lizard, snake, turtle
small mammals                   hamster, mouse, rabbit, shrew, squirrel
trees                           maple, oak, palm, pine, willow
vehicles 1                      bicycle, bus, motorcycle, pickup truck, train
vehicles 2                      lawn-mower, rocket, streetcar, tank, tractor
"""

from scipy.misc import toimage
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.models import model_from_json
from Utils import Dataset
import os
import pickle # "Pickling” is the process whereby a Python object hierarchy is converted into a byte stream:
_pickle_kwargs = {'encoding': 'latin1'}

batch_size = 32
num_classes = 100
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
JSON_model_name = "keras_cifar100_trained_model.json"
HDF5_model_name = "keras_cifar100_trained_model.h5"

print('Loading CIFAR-100 dataset with fine labels')

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# stored integers refer to alphabetically sorted fine labels
#sorting = np.argsort(cifar100_fine_labels)
#y_train = np.array([sorting[y] for y in y_train], dtype='uint8')
#y_test = np.array([sorting[y] for y in y_test], dtype='uint8')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
HDF5_model_path = os.path.join(save_dir, HDF5_model_name)
model.save(HDF5_model_path)
JSON_model_path = os.path.join(save_dir, JSON_model_name)
with open(JSON_model_path, "w") as json_file:
    json_file.write(model.to_json())
print("Saved model to disk")

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


