# ----import libraries
from tensorflow.keras.layers import Conv2D, Dropout,MaxPool2D
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

print(keras.__version__)
print(tf.__version__)

# ----Adding a path
trian_data_path = (r'D:\\machine learning practice\\Deep Learning\\Deep Learning Project\\Face Mask Detection\\dataset\\train')
validation_data_path = (r'D:\\machine learning practice\\Deep Learning\\Deep Learning Project\\Face Mask Detection\\dataset\\valid')



# ----Showing augmented images
def plotImages(image_arr):
    fig, axes = plt.subplots(1,5,figsize=(20,20))
    axes = axes.flatten()
    for img,ax in zip(image_arr,axes):
        ax.imshow(img)
    plt.tight_layout()
    # plt.show()

# ---this is the augmentation configuration we will use for training
# ---It generate more images using below parameters
training_datagen = ImageDataGenerator(rescale=1/255,rotation_range =40,width_shift_range=0.2,height_shift_range = 0.2,shear_range=0.2,zoom_range = 0.2,horizontal_flip=True,fill_mode='nearest')


# ----this is a generator that will read picture found in 
# ----at train_data_path,and indefinitely generate
# ----batches of augmented image data
training_data = training_datagen.flow_from_directory(trian_data_path, #this is the target directory
target_size = (200,200), #all images will be resized to 150X150
batch_size=128,
class_mode='binary' )#since we use binary_crossentropy loss,we need binary labels)

print(training_data.class_indices)

# ----this is the augmentation configration we will use for validation:
# ----only rescalling
valid_datagen = ImageDataGenerator(rescale= 1./255)

# ---this is a similar generator, for validation data
valid_data = valid_datagen.flow_from_directory(validation_data_path,target_size=(200,200),batch_size =128,class_mode = 'binary')


# ----Now showing augmented Images
images = [training_data[0][0][0] for i in range(5)]
plotImages(images)

# ----Save best model using vall accuracy
model_path = 'D:\machine learning practice\Deep Learning\Deep Learning Project\Face Mask Detection/model.last.h5'
checkpoint = ModelCheckpoint(model_path,monitor='val_accuracy',verbose = 1,save_best_only = True,mode = 'max')
callbacks_list = [checkpoint]

# ----Building CNN model
cnn_model = keras.models.Sequential([keras.layers.Conv2D(filters=32,kernel_size=5,input_shape=[200,200,3]),
keras.layers.MaxPooling2D(pool_size =(4,4)),
keras.layers.Conv2D(filters=64,kernel_size=4),
keras.layers.MaxPooling2D(pool_size=(3,3)),
keras.layers.Conv2D(filters=128,kernel_size=3),
keras.layers.MaxPooling2D(pool_size=(2,2)),
keras.layers.Conv2D(filters=256,kernel_size=2),
keras.layers.MaxPooling2D(pool_size=(2,2)),
keras.layers.Dropout(0.5),
keras.layers.Flatten(),#neural network builing
keras.layers.Dense(units=128,activation='relu'),#input layers
keras.layers.Dropout(0.1),
keras.layers.Dense(units=256, activation='relu'),
keras.layers.Dropout(0.25),
keras.layers.Dense(units=2,activation='softmax') #output layer
])


# ----compile CNN model
cnn_model.compile(optimizer = Adam(lr=0.001),loss='sparse_categorical_crossentropy',metrics = ['accuracy'])

history=cnn_model.fit(training_data,epochs=1,verbose=1,validation_data=valid_data,callbacks=callbacks_list)

# ----saving the model
cnn_model.save('D:\machine learning practice\Deep Learning\Deep Learning Project\Face Mask Detection/model.last.h5')

# ----Again training
history = cnn_model.fit(training_data,epochs=1,verbose = 1,validation_data=valid_data,callbacks=callbacks_list)

# ----for againy save the model
cnn_model.save('D:\machine learning practice\Deep Learning\Deep Learning Project\Face Mask Detection/model.last.h5')

# ----Now we plot the loss of the images---plot the loss
# plt.plot(cnn_model.history['loss'],label='train loss')
# plt.plot(cnn_model.history['val_loss'],label = 'val loss')
# plt.legend()
# plt.show()
# plt.savefig('LossVal_loss')

# ---Plotting the accuracy
# plt.plot(cnn_model.history['accuracy'],label = 'train acc')
# plt.plot(cnn_model.history['val_accuracy'],label='val acc')
# plt.legend()
# plt.show()
# plt.savefig('AccVal_acc')


