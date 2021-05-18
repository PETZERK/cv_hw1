import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.resnet import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D

norm_size = 64
EPOCHS = 8
INIT_LR = 1e-3
classnum = 80
CROP_SIZE = 299
batch_size = 8
seed = 1
train_data_dir = 'train'
validation_data_dir = 'val'
nb_train_samples = 20000
nb_validation_samples = 10000
imgList = []
labelList = []


def loadImageData():
    X_train = []
    Y_train = []
    img_train = []
    listImage = os.listdir(train_data_dir)
    for imgDir in listImage:
        labelName = imgDir
        labelList.append(labelName)
        imgDirPath = os.listdir(os.path.join(train_data_dir, imgDir))
        for img in imgDirPath:
            Y_train.append(int(imgDir))
            dataImgPath = os.path.join(train_data_dir, imgDir, img)
            print(dataImgPath)
            image = cv2.imread(dataImgPath)
            image = cv2.resize(image, (norm_size, norm_size))
            img_train.append(image)

            x = np.array(image)

            x = preprocess_input(x)
            X_train.append(x)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return X_train, Y_train


base_model = ResNet50(weights='imagenet', include_top=False)

new_layer = base_model.output
new_layer = GlobalAveragePooling2D()(new_layer)
new_layer = Dense(1024, activation='relu')(new_layer)

classes = Dense(80, activation='softmax')(new_layer)

model = Model(inputs=base_model.input, outputs=classes)

for layer in base_model.layers:
    layer.trainable = False

print("开始加载数据")
imgList, labelList = loadImageData()
labelList = np.array(labelList)
print("加载数据完成")
print(labelList)
trainX, valX, trainY, valY = train_test_split(imgList, labelList, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # 验证集不做图片增强

train_generator = train_datagen.flow(trainX, trainY, batch_size=batch_size, shuffle=True)
val_generator = val_datagen.flow(valX, valY, batch_size=batch_size, shuffle=True)
checkpointer = ModelCheckpoint(filepath='weights_best_Reset50_model.hdf5',
                               monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

reduce = ReduceLROnPlateau(monitor='val_accuracy', patience=10,
                           verbose=1,
                           factor=0.5,
                           min_lr=1e-6)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.save('my_model_resnet6.h5')

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

for layer in model.layers[:174]:
    layer.trainable = False
for layer in model.layers[174:]:
    layer.trainable = True

optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_generator,
                              steps_per_epoch=nb_train_samples // batch_size,
                              epochs=EPOCHS,
                              verbose=1,
                              validation_data=val_generator,
                              validation_steps=nb_validation_samples // batch_size,
                              max_queue_size=20,
                              shuffle=True,
                              workers=4)
print(history)

model.save('my_model_resnet6.h5')
