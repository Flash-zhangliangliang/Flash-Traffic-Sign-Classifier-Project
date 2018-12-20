#**Traffic Sign Recognition**

##Writeup

---


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data_set_distribution.png "Visualization"
[image2]: ./new_image/11.png "Traffic Sign 1"
[image3]: ./new_image/12.png "Traffic Sign 2"
[image4]: ./new_image/13.png "Traffic Sign 3"
[image5]: ./new_image/18.png "Traffic Sign 4"
[image6]: ./new_image/34.png "Traffic Sign 5"
[image7]: ./new_image/36.png "Traffic Sign 6"
[image8]: ./new_image/38.png "Traffic Sign 7"
[image9]: ./new_image/39.png "Traffic Sign 8"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/liruixuan-xidian/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_new.ipynb)

###Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the datadistribute.

![alt text][image1]

###Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because the experience shows that the gray level have much influence in image recognition. The gray image also have less dimension which could have less compute complexity.

As a second step, I normalized the image data so that I could speed up the gradient descent algorithm.


My final model consisted of the following layers:

|      Layer      |                Description                |
|:---------------:|:-----------------------------------------:|
|      Input      |            32x32x1 Gray image             |
| Convolution 5x5 | 1x1 stride, same padding, outputs 28x28x6 |
|      RELU       |                                           |
|   Max pooling   |       2x2 stride,  outputs 14x14x6        |
| Convolution 5x5 |       1x1 stride,  outputs 10x10x16       |
|      RELU       |                                           |
|   Max pooling   |        2x2 stride,  outputs 5x5x16        |
|     Flatten     |                Output 400                 |
|     Dropout     |               Keep_prob 0.5               |
| Fully connected |              Output 120                   |
|      RELU       |                                           |
| Dropout         | Keep_prob 0.5  |
| Fully connected |                 Output 84                 |
|      RELU       |                                           |
|     Dropout     | Keep_prob 0.5  |
| Fully connected |                 Output 43                 |


To train the model, I used an optimizer called tf.train.AdamOptimizer. The batch size has been set to 64 and number of epochs has been set to 50. Learning rate is 0.001.

My final model results were:
* validation set accuracy of 95.6%
* test set accuracy of 94.0%

according to the Udacity advice, I grayscalize and nomarlize the original data, when I use formulate x/255-0.5.
I have tred to change the value of batchsize,learning rate,epoch, but the accuracy is less than 93%. so I realize the model need to be modified.
and I add a dropout layer before fully connected layer. at last I get accuracy of 95.6% in validation set and accuracy of 94.0% in test set.


###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4]
![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9]

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Turn left ahead  		| Turn left ahead								|
| General caution		| General caution						    	|
| Keep right			| Keep right									|
| Yield	      		    | Yield					 				        |
| Right-of-way          | Right-of-way                                  |
| Keep left             | Keep left                          |
| Priority road         | Priority road                                 |
| Go straight or right  | Go straight or right                          |

The model was able to correctly guess 9 of the 9 traffic signs, which gives an accuracy of 100%!
