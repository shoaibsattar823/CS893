import os
import cv2
from imutils import paths
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt


'''LOADING DATA'''
data_dir = 'Cyclone_Wildfire_Flood_Earthquake_Database/'

data = []
labels = []

print('loading data...')
class_dirs = sorted(os.listdir(data_dir))
for direc in class_dirs:
    class_dir = os.path.join(data_dir, direc)
    for imagepath in sorted(list(paths.list_images(class_dir))):
        image = cv2.imread(imagepath)
        image = cv2.resize(image, (150, 150))
        data.append(image)
        labels.append(direc)

data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)


'''CREATING TEST TRAIN SPLITS'''
print('creating train test split...')
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=21)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)


'''DEFINING DATA GENERATOR TO LOAD DATA IN BATCHES'''
BS = 32
datagen = ImageDataGenerator()
train_gen = datagen.flow(trainX, trainY, batch_size=BS)


'''
DEFINING MODEL ARCHITECTURE
'''
print('creating model...')
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(4, activation='softmax'))
print(model.summary())


'''COMPILING THE MODEL'''
INIT_LR = 0.001
EPOCHS = 75
opt = optimizers.SGD(lr=INIT_LR)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


'''TRAINING'''
import math
# es = EarlyStopping(monitor='val_loss', mode='min', patience=20)
tb = keras.callbacks.TensorBoard(log_dir='logs/', histogram_freq=1)
print('training...')
train_size = len(trainX)
test_size = len(testX)
train_steps = math.ceil(train_size/BS)
valid_steps = math.ceil(test_size/BS)
H = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                        epochs=EPOCHS, validation_data=(testX, testY), callbacks=[tb])
model.save('my_disaster_classifier3.h5')


'''EVALUATING THE MODEL'''
from sklearn.metrics import classification_report
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))


'''PLOT OF ACCURACY AND LOSS WRT EPOCHS'''
N = np.arange(0, EPOCHS)
plt.style.use('ggplot')
plt.figure()
plt.plot(N, H.history['loss'], label='train_loss')
plt.plot(N, H.history['val_loss'], label='val_loss')
plt.plot(N, H.history['acc'], label='train_accuracy')
plt.plot(N, H.history['val_acc'], label='val_accuracy')
plt.title('Training loss and accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()


'''CONFUSION MATRIX'''
from sklearn.metrics import confusion_matrix
print(confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1)))