import os
from keras.layers import Dense, Conv2D , MaxPooling2D ,AveragePooling2D , Flatten, InputLayer
from keras.models import Sequential , load_model , Model
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam , RMSprop


model=Sequential()


model.add(Conv2D(filters=32,kernel_size=(4,4),strides=(2,2),input_shape=(64,64,3)))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(units=256,activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))

print(model.summary())

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '/task3/dataset/cnn/training_set/',
        target_size=(64, 64),
        batch_size=50,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        '/task3/dataset/cnn/test_set/',
    target_size=(64, 64),
        batch_size=50,
        class_mode='binary')




model.compile(optimizer=Adam(),loss='binary_crossentropy',metrics=['accuracy'])
model.fit(
        training_set,
        steps_per_epoch=8000/50,
        epochs=1,
        validation_data=test_set,
        validation_steps=900)
model.save('/task3/job3/cnn_model.hdf5')
accuracy=str(model.history.history['accuracy'][0]*100)
f=open('/task3/job3/cnn_accuracy.txt','w')

f.write(accuracy)
f.close()
model.save('cnn_model.hdf5',overwrite=True)
