import csv
import cv2
import numpy as np
import sklearn

# read log data
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) #skips the first line
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from sklearn.utils import shuffle

print("Number of training samples: ",len(train_samples))
print("Number of validation samples: ",len(validation_samples))

# load image from row/index
def load_image(index, sample):
    return cv2.imread('data/IMG/' + sample[index].split('/')[-1])

# flip image
def flip_input(image, angle):
    processed_image = cv2.flip(image,1)
    processed_angle = angle*-1.0
    return (processed_image, processed_angle)

 # generate images to save memory   
def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while(1):#loop while always true
        shuffle(samples)
        for offset in range(0, num_samples , batch_size):
            batch_samples =samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.25

            for batch_sample in batch_samples:

            	# load center image / angle
                center_image = load_image(0, batch_sample) 
                center_angle = float(batch_sample[3])

                # flip center image / angle
                center_flipped = flip_input(center_image, center_angle)
                images.extend([center_image, center_flipped[0]])
                angles.extend([center_angle, center_flipped[1]])
          
                # load left image / angle
                left_image = load_image(1, batch_sample)
                left_angle = center_angle + correction

                # flip left image /angle 
                left_flipped = flip_input(left_image, left_angle)
                images.extend([left_image, left_flipped[0]])
                angles.extend([left_angle, left_flipped[1]])

                # load right image / angle
                right_image = load_image(2, batch_sample)
                right_angle = center_angle - correction

                # flip right image / angle
                right_flipped = flip_input(right_image, right_angle)
                images.extend([right_image, right_flipped[0]])
                angles.extend([right_angle, right_flipped[1]])
            
            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train , y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers import Conv2D, Cropping2D, SpatialDropout2D
from keras.layers import MaxPooling2D

     
# Our model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()

# trim image to only see section with road
model.add(Cropping2D(cropping=((52,23), (0,0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation (Normalize)
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
#five convolutional and maxpooling layers

## three 5x5 convolution
model.add(Conv2D(24, (5, 5), padding="same", strides=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(36, (5, 5), padding="same", strides=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(48, (5, 5), padding="same", strides=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

## two 3x3 convolution
model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

#five fully connected layers
model.add(Dense(1164))
model.add(Activation('relu'))

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer ='adam') 

# Fit the model
history_object = model.fit_generator(train_generator, steps_per_epoch=(len(train_samples) / batch_size), validation_data=validation_generator, validation_steps=(len(validation_samples)/batch_size), epochs=10, verbose=1)

# Save model                   
model.save('model.h5')
print('Model saved successfully')
          
### print the keys contained in the history object
print ('History Keys')          
print(history_object.history.keys())

import matplotlib.pyplot as plt         
 
# Plot the training and validation loss for each epoch
print('Generating loss chart...')
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('model.png')

# Done
print('Done.')                    
