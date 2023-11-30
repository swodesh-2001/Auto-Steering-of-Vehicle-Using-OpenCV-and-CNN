**Auto Steering of Vehicle Using OpenCV and CNN**

**Project Overview**

This repository contains the code and documentation for the Auto Steering of Vehicle project, developed as part of the Computer Vision Semester Project for the FuseMachine AI Fellowship 2023.

**Team Member**

Swodesh Sharma

**Project Description**

This project is focused on implementing lane-following capabilities for autonomous vehicles in the Gym Donkey Car simulation environment. The system employs OpenCV to use image processing techniques to steer the car in the track. While the car is driven just by image processing algorithm, the data is collected to further train it in Convolutional Neural Networks (CNN) for steering angle prediction. The workflow involves reading data from the Gym Donkey Car simulation, applying image transformations, and utilizing computer vision techniques to extract lane information. And using these lane information , the car is steered in the track and the dataset is collected for Deep Learning.

**Workflow**

**Simulation Environment:**
Utilize Gym Donkey Car simulation for generating training and testing data.
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

![image](https://github.com/swodesh-2001/Auto-Steering-of-Vehicle-Using-OpenCV-and-CNN/assets/70265297/cea059a7-a727-4170-9d52-557e04990415)

5) Steering angle calculation
By iteratively moving this window along the vertical axis, lane line points are effectively captured throughout the entire image. Subsequently, a second-order polynomial is fitted to these identified lane points using techniques like polynomial regression. The coefficients of this polynomial model are then utilized to extract crucial information such as lane curvature and position. This information is essential for steering angle prediction, as it provides insights into the road geometry. The calculated steering angle is then transmitted to the Gym Donkey Car simulation, allowing for autonomous steering adjustments based on the real-time analysis of lane information.


