
import warnings
warnings.filterwarnings('ignore')
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


IMAGE_SIZE = [224, 224]

train_path = 'C:\\Users\\aisha user\\Desktop\\archive\\chest_xray\\chest_xray\\train'
valid_path = 'C:\\Users\\aisha user\\Desktop\\archive\\chest_xray\\chest_xray\\test'


vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


for layer in vgg.layers:
    layer.trainable = False

folders = glob('C:\\Users\\aisha user\\Desktop\\archive\\chest_xray\\chest_xray\\train\\*')
x = Flatten()(vgg.output)


prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

model.summary()

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory('C:\\Users\\aisha user\\Desktop\\archive\\chest_xray\\chest_xray\\train',
                                                 target_size=(224, 224),
                                                 batch_size=10,
                                                 class_mode='categorical')  


test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory('C:\\Users\\aisha user\\Desktop\\archive\\chest_xray\\chest_xray\\test',
                                            target_size=(224, 224),
                                            batch_size=10,
                                            class_mode='categorical')  

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


r = model.fit_generator(training_set,
                        validation_data=test_set,
                        epochs=1,
                        steps_per_epoch=len(training_set),
                        validation_steps=len(test_set))


import tensorflow as tf
from keras.models import load_model

model.save('C:/Users/aisha user/Downloads/Chest_x_ray_Detection-master (1)/chest_xray.h5')

from keras.models import load_model

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input


import numpy as np



model=load_model('C:/Users/aisha user/Downloads/Chest_x_ray_Detection-master (1)/chest_xray.h5')



img=image.load_img('C:\\Users\\aisha user\\Desktop\\archive\\chest_xray\\chest_xray\\test\\NORMAL\\IM-0022-0001.jpeg',target_size=(224,224))


x=image.img_to_array(img)


x=np.expand_dims(x, axis=0)



img_data=preprocess_input(x)



classes=model.predict(img_data)

result=int(classes[0][0])


if result==0:
    print("Person is Affected By PNEUMONIA")
else:
    print("Result is Normal")
