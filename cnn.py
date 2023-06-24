import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))
model.add(MaxPooling2D(2,2))
#model.add(Convolution2D(32,3,3, activation='relu'))
#model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tr_datagen = ImageDataGenerator(rescale=1/255, shear_range=0.2, zoom_range=0.2,
                                horizontal_flip=True)

ts_datagen = ImageDataGenerator(rescale=1/255)

tr_dataset = tr_datagen.flow_from_directory('dataset/training_set', target_size=(64,64),
                                            batch_size=32, class_mode='binary')

ts_dataset = ts_datagen.flow_from_directory('dataset/test_set', target_size=(64,64),
                                            batch_size=32, class_mode='binary')

model.fit(tr_dataset, steps_per_epoch=int(4000/32), epochs=2, validation_data=ts_dataset,
          validation_steps=int(1000/32))
#prediction
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image)

print(tr_dataset.class_indices)

if result[0][0] == 1:
    print("Dog")
else:
    print("Cat")
