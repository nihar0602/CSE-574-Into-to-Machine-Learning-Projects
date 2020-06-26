# CSE 574: Introduction to Machine Learning
<p align="center">
<img src="Project-1/MD Files/ub.png" alt="ub_logo.jpg" width="100" height="100">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="Project-1/MD Files/ub logo.png" alt="ub_log.jpg"> <br>

  <b> Course offered by Professor Varun Chandola in Spring 2020 </b>
</p>

### [Linear Models for Supervised Learning](https://github.com/nihar0602/CSE-574-Into-to-Machine-Learning-Projects/tree/master/Project-1) :
<img src="Project-1/MD Files/bar.jpg" alt="bar.jpg" width="1100" height="3"> <br>

**Problem:** 
`Linear Models for Supervised Learning` :The goal of this project is to write linear machine learning models from scratch. Models like Linear Regression, Logistic Regression, Support Vector Machine (SVM) and Perceptron Learning. Applying Minimizing techniques like direct and Gradient Descent Method and compare the results on the given DIABETES dataset. Use of any Python libraries/toolboxes, built-in functions, or external tools/libraries that directly perform classification, regression, function fitting, etc. was prohibited. 

Results: <br>

[Project Report](https://github.com/nihar0602/CSE-574-Into-to-Machine-Learning-Projects/blob/master/Project-1/Report.pdf)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

### [Non-Linear Models for Supervised Learning](https://github.com/nihar0602/CSE-574-Into-to-Machine-Learning-Projects/tree/master/Project-2) :
<img src="Project-1/MD Files/bar.jpg" alt="bar.jpg" width="1100" height="3"> <br>

**Problem:** 
`Non-Linear Models for Supervised Learning`: The goal of this project is divided into two parts. 

**Part-1:**`Sentiment Analysis`: In part-1, the goal is to develop the Machine Learning model which can classify movie reviews as positive or negative using two approaches. Word count vectorization and Neural Network approach are used to train the model. 

In word count vectorization, the approach was to iterate through each review, obtain the word count of each word and prepare two labels: Positive and Negative. The words are referred to be as vocabulary. Based on the word, we classify to its corresponding label. During testing, based on the word count for each feautre, it classifies it as positive or negative review. 

In Neural Network based approach, we applied different strategies. Strategies like increasing the width of the hidden layer, adding more hidden layers or increasing the number of epochs.Based on that a report was prepared which includes the accuracy on the test data set and the running time of the model. 

**Part-2:**`Image Classification on the AI Quick Draw Dataset`: The goal of this project is to study the performance of the multi-layered pereceptron (Neural Network) on the AI quick draw dataset. The task was performed on the some portion of the original quick draw dataset. The original dataset consist of 50 million drawings, while we trained model on 4 million drawings dataset. A report is created which shows detailed information and evaluation performance(both accuracy and time) of the model on the test data set.  

Results: <br>

[Project Report](https://github.com/nihar0602/CSE-574-Into-to-Machine-Learning-Projects/blob/master/Project-2/report.pdf)


### [Panorama/Image Stitching](https://github.com/nihar0602/CSE-573-Computer-Vision-and-Image-Processing--Projects/tree/master/Project%20-2) :
<img src="Project-1/MD Files/bar.jpg" alt="bar.jpg" width="1100" height="3"> <br>

**Problem:** 
`Image Stitching`: Create a panoramic image from at most 5 images. The goal of this project is to experiment with image stitching methods. Given a set of photos, your
program should be able to stitch them into a panoramic photo. Overlap of the given images will be at least 20%. Any API provided by OpenCV could be used, except “`cv2.findHomography()`” and APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “`cv2.BFMatcher()`” and “`cv2.Stitcher.create()`”.

**Approach:**
- Keypoints detection and 128 bit feature vector computation using `SIFT` descriptor. 
- Created an algorithm that can define the order of the images if given in randomized order.
- Homography matrix generation using `SVD` technique.
- Implemented `RANSAC` algorithm for finding the best Homography matrix
- Stitched all images


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

**Results:** 

Output image: <br>
<img src="Project -2/data/panorama.jpg" alt="panoroma.jpg">
<img src="Project -2/extra1/panorama.jpg" alt="panoroma.jpg">
<img src="Project -2/extra3/panorama.jpg" alt="panoroma.jpg">



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

### [Face Detection in the Wild using Viola Jones Algorithm](https://github.com/nihar0602/CSE-573-Computer-Vision-and-Image-Processing--Projects/tree/master/Project%20-3) :
<img src="Project-1/MD Files/bar.jpg" alt="bar.jpg" width="1100" height="3"> <br>

**Problem:** 
The goal of this project is to implement the `Viola-Jones` face detection algorithm which is capable of detecting frontal faces in real time and is regarded as a milestone in the development of computer vision. Given a face detection dataset composed of thousands of images, the goal is to train a face detector
using the images in the dataset. The trained detector should be able to locate all the faces in any image coming from the same distribution as the images in the dataset. Any APIs provided by OpenCV that have “cascade”, “Cascade”, “haar” or “Haar” functionality can not be used. Using any APIs that implement part of Viola-Jones algorithm directly, e.g., an API that computes integral image, will result in a deduction of 10% − 100% of the maximum possible points of this project

**Approach:**
- Used `FDDB` dataset to train the model with 'face images and 'non-face images'.
- Implemented `integral image` for the feature extraction. 
- Implemented `Adaboost` algorithm to extract best features that can detect faces. 
- Trained the data set using Google Cloud Platform. 
- An attempt on `CASCADE` algorithm was made to reject non-face region quickly


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Project Report](https://github.com/nihar0602/CSE-573-Computer-Vision-and-Image-Processing--Projects/blob/master/Project%20-3/Report.pdf)


**Output:** <br>

<img src="Project -3/Results/827.jpg" width="400" height="300" align="left"><img src="Project -3/Results/898.jpg" width="400" height="300" align="left">
<img src="Project -3/Results/903.jpg" width="400" height="300" align="left"><img src="Project -3/Results/932.jpg" width="400" height="300" align="left">

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

