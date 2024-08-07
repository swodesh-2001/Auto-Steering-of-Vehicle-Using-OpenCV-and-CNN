<h1>Auto Steering of Vehicle Using OpenCV and CNN </h1>

**Project Overview**

This repository contains the code and documentation for the Auto Steering of Vehicle project, developed as part of the Computer Vision Semester Project for the FuseMachine AI Fellowship 2023.

**Team Member**

Swodesh Sharma

**Project Description**

This project is focused on implementing lane-following capabilities for autonomous vehicles in the Gym Donkey Car simulation environment. The system employs OpenCV to use image processing techniques to steer the car in the track. While the car is driven just by image processing algorithm, the data is collected to further train it in Convolutional Neural Networks (CNN) for steering angle prediction. The workflow involves reading data from the Gym Donkey Car simulation, applying image transformations, and utilizing computer vision techniques to extract lane information. And using these lane information , the car is steered in the track and the dataset is collected for Deep Learning.

<h2>Workflow</h2>

 <h3>PART 1 : COMPUTER VISION </h3>

**Simulation Environment:**
Utilize Gym Donkey Car simulation for generating training and testing data.

Simulator Link : https://github.com/tawnkramer/gym-donkeycar

![image](https://github.com/swodesh-2001/Auto-Steering-of-Vehicle-Using-OpenCV-and-CNN/assets/70265297/1fcf24ac-fb8f-4962-9e81-4ceba34a8b8f)

**Image Processing with OpenCV**

Image from the donkeycar is read in Opencv and following techniques are applied.

1) Calibration

   The calibration function helps to get the values for HSV masking and warp points for Birds Eye view
    ![image](https://github.com/swodesh-2001/Auto-Steering-of-Vehicle-Using-OpenCV-and-CNN/assets/70265297/40abe175-bf21-4a2e-9fee-e093d228e233)

2) Using the values obtained from the calibration we perform , warp transformation and HSV masking.
3) Now once we get the HSV mask of the lane lines , we create a histogram of the image , to seperate the left and right lane lines
   
   ![image](https://github.com/swodesh-2001/Auto-Steering-of-Vehicle-Using-OpenCV-and-CNN/assets/70265297/89d3e478-90cd-44e8-88d5-5a8efba90d03)
   
4)Lane Points Extraction:

After obtaining the left and right bases, a subsequent step involves constructing a small box around each of these bases and extracting the contour of the content enclosed within these boxes. Subsequently, a new set of boxes is generated, wherein the bases are determined by the centers of the contours identified in the preceding step. This iterative process allows for the creation of successive layers of boxes, each centered around the contours of the previous base, facilitating a refined and detailed analysis of the underlying content.

![image](https://github.com/swodesh-2001/Auto-Steering-of-Vehicle-Using-OpenCV-and-CNN/assets/70265297/467f3ce7-bd7a-4e72-92fa-0ba4c5e3b9a8)

5) Steering angle calculation
By iteratively moving this window along the vertical axis, lane line points are effectively captured throughout the entire image. Subsequently, a second-order polynomial is fitted to these identified lane points using techniques like polynomial regression. The coefficients of this polynomial model are then utilized to extract crucial information such as lane curvature and position. This information is essential for steering angle prediction, as it provides insights into the road geometry. The calculated steering angle is then transmitted to the Gym Donkey Car simulation, allowing for autonomous steering adjustments based on the real-time analysis of lane information.

![2023-11-30 21-38-22](https://github.com/swodesh-2001/Auto-Steering-of-Vehicle-Using-OpenCV-and-CNN/assets/70265297/d4ba5953-b658-47b7-ba86-537f68510e0d)

<h3>PART 2 : DEEP LEARNING </h3>

1) Data Collection,Balancing and Augmentation
Using the computer vision algorithm developed above, the car is driven around the track, and data is collected. This dataset comprises track images paired with their corresponding steering angles. This methodology is commonly referred to as Behavior Cloning.
In Behavior Cloning, the algorithm learns to mimic the driving behavior observed during data collection. The model is trained on the collected dataset, enabling it to generalize and autonomously navigate the track based on the learned steering patterns. This approach is widely utilized in the development of autonomous vehicles, leveraging the principles of supervised learning to emulate human-like driving skills.

a)Data Balancing

The car being driven has more data with 0 as the steering, which will make the model biased towards predicting 0 so we balance the data.
![image](https://github.com/swodesh-2001/Auto-Steering-of-Vehicle-Using-OpenCV-and-CNN/assets/70265297/677c6041-85b6-4b1f-96bb-c34468ca77af)

b)Data Augmentation.

The data was augmented with the ’imgaug’ augmentation library and applied only to the training data.

Data Augmentation includes
Zooming  allows the model to get closer look of some features

Image Panning  translation shifting

Altering Brightness  changes the brightness ( brighter or darker)

![image](https://github.com/swodesh-2001/Auto-Steering-of-Vehicle-Using-OpenCV-and-CNN/assets/70265297/d40b1120-beec-4928-b248-d3f3dd91925a)

These techniques make the data more versatile, allowing the model to generalize for various scenarios despite the limited amount of data 

c)RGB to YUV
This conversion helps to highlight the lane features of the track

 ![image](https://github.com/swodesh-2001/Auto-Steering-of-Vehicle-Using-OpenCV-and-CNN/assets/70265297/ac231fb9-df00-4ef0-a027-16b27e17c1c5)


Finally, a Gaussian blur is applied, which smooths the images and removes disturbances. 
The final steps in image processing are resizing and normalization, which resizes the image to the desired shape in the range [0, 1] helping speed up the training process.


3) Training
   
Model Architecture used for CNN.
![image](https://github.com/swodesh-2001/Auto-Steering-of-Vehicle-Using-OpenCV-and-CNN/assets/70265297/a3e5d299-754c-48cf-b188-6907c016b57b)

 Demo using CNN algorithm
 
![CNN](https://github.com/swodesh-2001/Auto-Steering-of-Vehicle-Using-OpenCV-and-CNN/assets/70265297/fc07a74b-ceb7-4cd8-9223-48ab60c6a0ce)


### Hardware Development


![Hardware](https://github.com/user-attachments/assets/6e405b7b-1668-421f-b6f2-7278d756581a)

