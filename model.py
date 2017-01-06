import pandas as pd
from PIL import Image
import os
import numpy as np
import cv2
from pathlib import Path

df = pd.read_csv("driving_log.csv")

#store features and labels in arrays
images = df[[0]]
images = images.values

images_left = df[[1]]
images_left = images_left.values

images_right = df[[2]]
images_right = images_right.values

steering = df[[3]]
steering = steering.values

OFF_CENTER_IMG = 0.25

row, col, ch = 66, 200, 3

##generator to yield images
def generate(BATCH):
    while (1):
        steering_angle = np.zeros(shape= (BATCH, 1), dtype="float32")

        image_list = np.zeros(shape = (BATCH, row, col, ch), dtype="float32")
        
        count = BATCH
        
        loc = 0
        while (count > 0):
            #random index
            i = np.random.randint(len(images))
            
            steering_angle[loc] = steering[i].item()

            ########randomly select center left and right images###############
            img_choice = np.random.randint(3)

            #left image with +0.25 steering angle offset
            if img_choice == 0:
                temp = str(images_left[i].item())
                
                my_file = Path(temp.strip())
                if not my_file.is_file():
                    continue
                    
                im = cv2.imread(temp.strip()) # left images have a space before the path

                steering_angle[loc] += OFF_CENTER_IMG

            #center image
            elif img_choice == 1:
                temp = str(images[i].item())
                
                my_file = Path(temp)
                if not my_file.is_file():
                    continue
                    
                im = cv2.imread(temp.strip())

            #right image with -0.25 steering angle offset
            else:
                temp = str(images_right[i].item())
                
                my_file = Path(temp.strip())
                if not my_file.is_file():
                    continue
                
                im = cv2.imread(temp.strip())

                steering_angle[loc] -= OFF_CENTER_IMG

            #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            ######## image transform for model ##########################
            im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
            im = im[ 64:295,:, :]
            im = cv2.resize(im, (col, row), interpolation=cv2.INTER_AREA)

            ##############randomly change brightness#################
            if np.random.randint(2) == 0:
                temp = cv2.cvtColor(im, cv2.COLOR_YUV2RGB)
                temp = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
                # Compute a random brightness value and apply to the image
                brightness = 0.25 + np.random.uniform()
                temp[:, :, 2] = temp[:, :, 2] * brightness
                
                im = cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)
                im = cv2.cvtColor(temp, cv2.COLOR_RGB2YUV)

            ########Flip images randomly################################
            if np.random.randint(2) == 0:  
                im = cv2.flip(im, 1)
                steering_angle[loc] = -steering_angle[loc]
            
            #############################################################
            ## X-axis and Y-axis translation
#             TRANS_X_RANGE = 100  # Number of translation pixels up to in the X direction for augmented data (-RANGE/2, RANGE/2)
#             TRANS_Y_RANGE = 40  # Number of translation pixels up to in the Y direction for augmented data (-RANGE/2, RANGE/2)
#             TRANS_ANGLE = .3  # Maximum angle change when translating in the X direction
            
#             # Randomly form the X translation distance and compute the resulting steering angle change
#             if np.random.randint(2) == 0:
#                 x_translation = (TRANS_X_RANGE * np.random.uniform()) - (TRANS_X_RANGE / 2)
#                 steering_angle[loc] += ((x_translation / TRANS_X_RANGE) * 2) * TRANS_ANGLE

#                 # Randomly compute a Y translation
#                 y_translation = (TRANS_Y_RANGE * np.random.uniform()) - (TRANS_Y_RANGE / 2)

#                 # Form the translation matrix
#                 translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])

#                 # Translate the image
#                 im = cv2.warpAffine(im, translation_matrix, (im.shape[1], im.shape[0]))

            ###########################################################
            
            image_list[loc] = im            
            
            count -= 1
            loc += 1                

        yield image_list, steering_angle


train_generator = generate(256)
valid_generator = generate(20)


###NVIDIA Arch

import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

model = Sequential()

#model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(row, col, ch)))

model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))
#model.add(ELU())
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))
#model.add(ELU())
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))
#model.add(ELU())
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="he_normal"))
#model.add(ELU())
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="he_normal"))

model.add(Flatten())

model.add(Activation('relu'))
#model.add(ELU())

# model.add(Dense(1164, init="he_normal"))
# model.add(Activation('relu'))
#model.add(ELU())
model.add(Dense(100, init="he_normal"))
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dense(50, init="he_normal"))
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dense(10, init="he_normal"))
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dense(1, init="he_normal"))

model.summary()

##compile and train the model
from keras.optimizers import Adam

#adam = Adam(lr=1e-4)
model.compile(loss="mean_squared_error", optimizer="adam")

#model.fit_generator(data_generator, samples_per_epoch=20000, nb_epoch=2)
model.fit_generator(train_generator, samples_per_epoch=20224, nb_epoch=15, validation_data = valid_generator, nb_val_samples=1000)

##save arch as json and model weights
import json
json_string = model.to_json()
json.dump(json_string, open("model.json", "w"))
model.save_weights('model.h5')