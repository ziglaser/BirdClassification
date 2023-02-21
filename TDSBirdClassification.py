import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras import optimizers, applications
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense, LeakyReLU
from keras.utils import to_categorical
import math
import PIL
import os
import shutil
import imgaug as ia
import imgaug.augmenters as iaa
import imageio as io



# CONSTANTS
# img constraints
img_width, img_height = 224, 224

# file paths
top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = "/mnt/c/Users/zglas/Pictures/BirdProject/train"
validation_data_dir = "/mnt/c/Users/zglas/Pictures/BirdProject/validation"

# model training specs
epochs = 15
batch_size = 8

# RUN PARAMETERS
guidance1 = input("Acquire Data? (y/n) ")
data_acquisition = guidance1 in ['yes', 'y', 'Yes', 'Y']

guidance2 = input("Train Model? (y/n) ")
train_model = guidance2 in ['yes', 'y', 'Yes', 'Y']

# TRANSFER MODEL
# load vgc16 model
vgg16 = applications.VGG16(include_top=False, weights='imagenet')
datagen = ImageDataGenerator(rescale=1./255)

class DataSet:
    def __init__(self, directory, mode):
        self.directory = directory
        self.mode = mode
        self.generator = None
        self.bottleneck = None
        self.num_classes = None
        self.labels = None
        self.predictions = None

    def create_generator(self):
        self.generator = datagen.flow_from_directory(self.directory,
                         target_size=(img_width, img_height),
                         batch_size=batch_size,
                         class_mode='categorical',
                         shuffle=False)

    def create_bottleneck(self):
        if data_acquisition or self.mode == 'test':
            nb_train_samples = len(self.generator.filenames)
            print(nb_train_samples)
            predict_size = int(math.ceil(nb_train_samples / batch_size))
            bottleneck_features = vgg16.predict(self.generator, predict_size)
            np.save('bottleneck_features_' + self.mode + ".npy", bottleneck_features)
            self.bottleneck = bottleneck_features
        else:
            self.bottleneck = np.load('bottleneck_features_' + self.mode + ".npy")

    def predict(self):
        self.predictions = model.predict(self.bottleneck)

    def main(self):
        self.create_generator()
        self.create_bottleneck()
        self.num_classes = len(self.generator.class_indices)
        self.labels = to_categorical(self.generator.classes, num_classes=self.num_classes)


def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(100, activation=LeakyReLU(alpha=0.3)))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation=LeakyReLU(alpha=0.3)))
    model.add(Dropout(0.3))
    model.add(Dense(output_shape, activation='softmax'))
    return model

training_data = DataSet(train_data_dir, 'train')
training_data.main()

validation_data = DataSet(validation_data_dir, 'validation')
validation_data.main()

# The model
model = create_model(training_data.bottleneck.shape[1:], training_data.num_classes)
if train_model:
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    history = model.fit(training_data.bottleneck, training_data.labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data.bottleneck, validation_data.labels))
    model.save_weights(top_model_weights_path)
    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data.bottleneck, validation_data.labels, batch_size=batch_size, verbose=1)
else:
    model.load_weights(top_model_weights_path)


'''labels = (val_generator.class_indices)
labels = dict((v, k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]'''

img_root = "/mnt/c/Users/zglas/Pictures/BirdProject"
working_img_path = img_root + "/working_image/img"
try:
    shutil.rmtree(working_img_path)
except FileNotFoundError:
    pass
os.mkdir(working_img_path)
shutil.copyfile(img_root + "/temp/Gray_Catbird_0028_20598.jpg", working_img_path + "/original.jpg")

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.AdditiveGaussianNoise(scale=(10, 60)),
    # iaa.Crop(percent=(0, 0.2)),
    # iaa.Affine(rotate=(-25, 25)),
    ])


image = io.imread(working_img_path + "/" + os.listdir(working_img_path)[0])
images = [image] * 10
images_aug = seq(images=images)

for i in range(1, 11):
    io.imwrite(working_img_path + "/aug_" + str(i) + ".jpg", images_aug[i - 1])

test_generator = DataSet(img_root + "/working_image", 'test')
test_generator.main()
test_generator.predict()
print(test_generator.predictions)
