import numpy as np
import cv2
import os

from numpy import argmax
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.resnet import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

norm_size = 64
model = load_model('my_model_resnet4_0.3382.h5')
# model = ResNet50(weights='imagenet', classes=80, include_top=False)
validation_data_dir = 'test'
print('models loaded!')

X_val = []
listImage = os.listdir(validation_data_dir)
listImage.sort(key=lambda x: int(x[:-4]))
with open("181250180.txt", 'w') as file_object:
    for img in listImage:
        dataImgPath = os.path.join(validation_data_dir, img)
        print(dataImgPath)
        image = cv2.imread(dataImgPath)
        image = cv2.resize(image, (224, 224))
        image = image.reshape(1, 224, 224, 3)
        prediction = model.predict(image)
        print(img)
        print(np.argmax(prediction, axis=1)[0])
        file_object.write(img+' '+str(np.argmax(prediction, axis=1)[0])+'\n')
