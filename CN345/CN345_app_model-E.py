
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os

def image_gen_w_aug(train_parent_directory, test_parent_directory):
    
    train_datagen = ImageDataGenerator(rescale=1/255, #scaling
                                      rotation_range = 30,   #rotation 
                                      zoom_range = 0.2, 
                                      width_shift_range=0.1,  
                                      height_shift_range=0.1,
                                      validation_split = 0.15) #split to 15% of validation set.
    
    test_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                       target_size = (75,75),
                                                       batch_size = 100,
                                                       class_mode = 'categorical',
                                                       subset='training')
    
    val_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                          target_size = (75,75),
                                                          batch_size = 20,
                                                          class_mode = 'categorical',
                                                          subset = 'validation')
    
    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                     target_size=(75,75),
                                                     batch_size = 5, #call by batches of 37
                                                     class_mode = 'categorical')
    
    return train_generator, val_generator, test_generator


def model_output_for_TL (pre_trained_model, last_output):

    x = Flatten()(last_output)
    
    # Dense hidden layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output neuron. 
    x = Dense(3, activation='softmax')(x) #using softmax.
    
    model = Model(pre_trained_model.input, x)
    
    return model


train_dir = os.path.join('δεδομένα/train') #change to ur own directory.
test_dir = os.path.join('δεδομένα/test') #change to ur own directory.
#using windows commans to set to the default path. --> Command prompt.

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir)

pre_trained_model = InceptionV3(input_shape = (75, 75, 3), 
                                include_top = False, 
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed3')
last_output = last_layer.output  #last layer is output.

model_TL = model_output_for_TL(pre_trained_model, last_output)
model_TL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

 #hyperparameters.
 #epoch is the number of training rounds
history_TL = model_TL.fit(
train_generator,
steps_per_epoch=10,
epochs=10,
verbose=2,
validation_data = validation_generator)

tf.keras.models.save_model(model_TL,'Auntie-DOT_CN345.hdf5') #will be save as this file


#HALO LORE
#Auntie DOT is UNSC's (dumb) AI used by NOBLE team during the fall of REACH, it is one of the AI's used by ONI before the discovery of forerunner tech which allowed humanity to create (smart) AI such as CORTANA.