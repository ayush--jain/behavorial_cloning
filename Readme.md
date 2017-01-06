# CarND-Behavioral-Cloning
My behavioral cloning project from Udactiy's Self-Driving Car Nanodegree.

## Data
I collected data provided by Udacity. I collected more data around tricky corners since most of the tracks were straight or soft curves.

I preprocessed the data for training by resizing to 66x200 pixel images and converting to YUV color scale, for using in the NVIDIA model. I also cropped the bottom 25 pix and top 10% of the image to remove unwanted noise. I used a generator to read images, perform the preprocessing and randomly augment the images and yield images and steering angles. For augmentation, I flipped random images horizontally and changed the brightness of the images randomly. I didn't create a different testing set because the real testing could be done by running the simulator in autonomous mode to get qualitative results.

## Model
For the model architecture I chose the [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). It consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. The convolutional layers are designed to perform feature extraction, and are chosen empirically through a series of experiments that vary layer configurations. We then use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers. A fully connected layer of 1 neuron was used at the end to output the steering angles. I used "relu" activations between the layers. Using dropout gave poor results because the model is already very trim.

I optimized the model with an Adam optimizer over MSE loss.

## Hyperparameters
I used the default parameters with Adam optimizer. I trained for 15 epochs and a batch size of 256 for training and 20 for validation. The number of samples per epoch for training was 20224 and 1000 for validation. 

## Results
Since the generator was randomly picking images, the training and validation accuracy were not consistent. Both training and validation accuracy were in the range of 0.02. The model can be evaluated qualitatively on the track where it drives the car well on track 1 without ever crashing or venturing into dangerous areas. However, it fails to generalize on track 2, possibly because it hasn't been trained for darker environments and slopes.

## Sample Images from camera view

#### Right image
![alt text](/images/right_image.png "Right image")

#### Center image 
![alt text](/images/image_center.png "Center image")

#### Left image
![alt text](/images/image_left.png "Left image")
