# CS4641_ML_Project

## Proposal Video
https://www.youtube.com/watch?v=NpODfgFbCEo&ab_channel=JiminSun

## Introduction / Background
The topic of our project is related to diabetic retinopathy (DR) detection for which we are going to train a deep learning model that is able to identify signs of diabetic retinopathy in eye images. Diabetic retinopathy is the most common diabetic eye disease and a leading cause of blindness in American adults. The retina is the light-sensitive tissue at the back of the eye. A healthy retina is necessary for good vision. According to the US national eye institute, from 2010 to 2050, the number of Americans with diabetic retinopathy is expected to nearly double, from 7.7 million to 14.6 million. As the number of people being diagnosed with diabetic retinopathy increases, it is necessary to construct an automated image classification model to automatically classify retinal tissue into healthy and pathological in early stages.

## Problem Definition
While taking active care in diabetes management can prevent diabetic retinopathy, with more than 1 in 10 people in the United States with diabetes, it is no doubt that diabetic retinopathy is also a prevalent complication that needs attention (Mayo Clinic, 2021). With the advanced technology and the ampleness of patient/non-patient data, our goal in this project is to use classification models using real data, to predict and diagnose diabetic retinopathy in patients.

## Data Collection
We used data from the Diabetic Retinopathy Detection dataset from Kaggle. The dataset consisted of a total of 35,127 training images, which were JPEG files of left and right eyes. There were a total of five classes (No DR, Mild, Moderate, Severe, Proliferate DR) within the training set. Because of limited resources to train the deep learning model, we took a random sample from each class to avoid data imbalance. Specifically, the training set consists of 560 images from each class, which sums up to 2800 images in total. Similarly, for the validation set, we have 140 images from each class (700 images in total) and for the test set, we have 100 images from each class (500 images in total). In percentage terms, the breakdown is 70% training, 17.5% validation, and 12.5% testing. All images were 3-channel colored images but had differing sizes. 

## Methods
### Image preprocessing
Before feeding the data into the model, we cleaned up our image data by performing a series of image preprocessing steps. First, a cropping method was run to make the images a uniform size of 256x256 pixels. Below is an example of the pre-cropped image and the cropped image of the eye.

![1](https://github.gatech.edu/storage/user/54998/files/498b2c85-fd62-4a2f-a940-18d59b046b69)
![2](https://github.gatech.edu/storage/user/54998/files/af324908-1535-401a-8443-482b348ccac7)<br/>
***image above: original image (left) and cropped image (right).***

Additionally, a normalization function, ranging from 0 to 1, was run to adjust the lighting of the images. Below are images of the non-normalized data and the normalized data.

![3](https://github.gatech.edu/storage/user/54998/files/14bd3b74-60ae-4e86-afad-8412d6dc0f5e)
![4](https://github.gatech.edu/storage/user/54998/files/cd65d4e0-bcd4-45d5-bb5b-042211083564)<br/>
***image above: cropped image (left) and cropped AND normalized image (right).***

The ‘after normalization’ image visually looks completely black since the color values were adjusted from range [0, 255] to [0, 1] for each red, green, and blue values. The human eye would not be able to distinguish the differences; however, when the model reads the dataset, it would distinguish and read the scaled down values of the normalized data.

Below is a histogram of ‘16_right.jpeg’ containing the pixel values for the pixels in the image:<br/>

![5](https://github.gatech.edu/storage/user/54998/files/2d8abb6f-45e3-4074-8977-04885a487abe)
![6](https://github.gatech.edu/storage/user/54998/files/e021e87e-93b3-41c4-ab73-69f3d951eb9d)<br/>
***Figure 1. Histograms of pixel values in non-normalized image (left) and normalized image (right).***

To verify that the normalization of the images helps in accurately training the models, we decided to run both the non-normalized and the normalized dataset in our deep learning models and compare their performance.

### Deep learning model - supervised approach
**Model initialization**: A modified VGG-16/ResNet18 model pre-trained on ImageNet was used for our 5-category classification in a transfer learning approach. Since the model was pre-trained on more than a million images from the ImageNet database, we believe that it carries strong ability to extract generic features from the dataset. To save computation time, we froze all weights except for the last layer in the model. To modify and adapt the pretrained VGG-16/ResNet18 model to our diabetic retinopathy detection project, we first printed out and examined the architecture of the model using the .eval() method, then substituted the last fully-connected layer and changed the out_features variable from the original 1,000 to 5 to achieve a five-category output. We have noticed that VGG16 and ResNet 18 have different layouts in terms of architecture, so we have modified our code accordingly.

**Generic model class**: To lay out the pipeline more clearly, we have created a generic model class that takes in the modified model, the train DataLoader, the test DataLoader, the criterion (loss function), and the optimizer, and outputs the running loss and the accuracy. 

**Model training**: After we initialized and modified the VGG16 model, we determined the criterion and optimizer that would be used during our model training process. We used cross entropy as the loss function and stochastic gradient descent (SGD) as the optimizer for our model. The optimizer was set to have only the parameters of the classifiers being optimized. We trained the model using the following hyperparameters: training lasted for 100 epochs with batch size = 56, learning rate = 0.001, momentum = 0.9, and weight decay = 0.0005.

### Performance metrics
**Accuracy**: the first metric that comes to our mind is the accuracy score, which is the percentage of predicted labels that match the true labels exactly. To calculate this metric, we used the accuracy_score() function from the sklearn.metrics package.

**Confusion matrix**: further, since our model is a multi-class classification model, we also looked at the confusion matrix to see how well the model performs on the test data. This gives us more information than a single accuracy score: if the predicted labels were wrong, we could see which classes the images were misclassified into. We used the confusion_matrix() function from the sklearn.metrics package. In the plot below, the rows are the true labels, and the columns are predicted labels. Label 0 means no diabetic retinopathy and label 5 means proliferative diabetic retinopathy.

**Customized accuracy measure**: the accuracy score only tells us whether the predictions match with the true labels exactly. However, for a misclassified data point, we also want to know how far away is the predicted label from the true label. If the prediction is only 1 class away from the ground truth, this is better if it is 4 classes away from the ground truth. Therefore, we developed a customized accuracy measure (average distance in the plots below) to represent this logic.

**AUC, MCC**: we also looked into AUC and MCC, also using the respective functions from the sklearn package, to measure the effectiveness of the model to gain a better understanding of the misclassifications. The Matthews correlation coefficient (MCC) gives a number between -1 and 1 to summarize the confusion matrix, and the area under the curve graph (AUC) measures the model’s accuracy between 0 and 1, where 1 means the model fitted everything correctly and 0 means the model fitted entirely incorrectly.

## Results and Discussion
### Performance of each model with two datasets
Two deep learning models (VGG16 and ResNet18) were trained with two different datasets. The first dataset (we call it non-normalized in later parts) was sized uniformly to 256x256 pixels but the pixel values were not normalized. The second dataset, in addition to resizing, normalized the pixel values into [0,1] (we call it normalized dataset in later parts). At the end, we wanted to compare the performance of two models as well as the effect of normalization. Ultimately, there were four experiments conducted: VGG 16 trained with non-normalized dataset, VGG 16 trained with normalized dataset, ResNet 18 trained with non-normalized dataset, and ResNet 18 trained with normalized dataset. Figure 2, 3, 4, 5 shows the metrics of the training and validation dataset after each of the 100 epochs for the four experiments.
### Test results of each experiments
Table 1 shows the test accuracy for the four experiments. Figure 6, 7, 8, 9 shows the confusion matrix for the four experiments. Each row of the confusion matrix indicates the true label, and the numbers sum up to 2. While each column of the confusion matrix indicates the predicted label. The color bar on the right indicates that as the color brightens (top part of the color bar), more data was assigned to the particular cell and vice versa.
### Analysis of test results
As Table 1 indicates, the best performing model is VGG 16 when trained with non-normalized images. Comparing this with the result of VGG 16 trained with normalized images may indicate that normalizing may have induced loss of information. However, the effect of normalization of pixel values were not clearly visible when we compared the two ResNet 18 models. This will be further looked into by investigating other possible factors such as difference in model structure. Overfitting occurred in all four experiments. While the training accuracy increased over epochs, the validation accuracy converged after several epochs. This may be due to the absence of data augmentation techniques. Data augmentation increases the amount of training data. Other techniques such as dropout or batch normalization could be potentially used to reduce the problem of overfitting. Figure 6, 7, 8, 9 indicates a confusion matrix of the test results. It can be seen that the bottom right corner (true label: class 4, predicted label: class 4) was the brightest or one of the brightest cells in the confusion matrix. From this, we arrived at the naive conjecture that the most severe diabetic retinopathy had the most distinguishable features among all severities. Overall, our four experiments showed a low performance. This may have been due to inconsistency of the images. Although we preprocessed the image to be of uniform shape, because the retinal fundus images were inherently different in shape and sizes, cropping from the center may have not solved the problem. Rather, cropping from the center may have induced data loss for some images by cropping off peripherals of the fundus. In addition, because we did not align where the optic disc was located, this might have caused further confusion of the model. However, this was a problem that could not be solved without a proper optic disc detection algorithm, because the image label did not match the position of the optic disc. Therefore, simply flipping the images based on the file name did not solve the problem. We also found that there might have been a problem when saving the images after normalizing the pixel values to [0,1] in JPEG format by looking at the histograms in Figure 1. This needs further investigation and we might need to reconsider the normalization techniques, such as subtracting the minimum pixel value from the image and dividing with the maximum pixel value, followed by multiplying it with 255. 

## Appendix
![7](https://github.gatech.edu/storage/user/54998/files/93d4d371-9fca-412e-aea2-34bf99d7ad58)
![8](https://github.gatech.edu/storage/user/54998/files/75c5e801-c0f7-4532-b8b3-86a75f6c9a61)
![9](https://github.gatech.edu/storage/user/54998/files/353c5438-28c5-4c40-937c-91714798148d)<br/>
***Figure 2. Model accuracy, average distance, and loss graph for VGG 16 trained with non-normalized dataset.***

![10](https://github.gatech.edu/storage/user/54998/files/cf7676d7-fdee-4fdf-a53b-34525c33bc89)
![11](https://github.gatech.edu/storage/user/54998/files/cce877ad-43c1-4bad-a488-acf361315b2c)
![12](https://github.gatech.edu/storage/user/54998/files/b55f525b-ebb0-4b4a-a913-f3aa6331cd83)<br/>
***Figure 3. Model accuracy, average distance, and loss graph for VGG 16 trained with normalized dataset.***

![13](https://github.gatech.edu/storage/user/54998/files/c91e925b-8e98-4588-be6f-eb4cb660be5a)
![14](https://github.gatech.edu/storage/user/54998/files/987f8588-64a7-4b9c-bbe4-3a44a690a0ff)
![15](https://github.gatech.edu/storage/user/54998/files/b7a2d04d-7286-44df-9175-d9257d349942)<br/>
***Figure 4. Model accuracy, average distance, and loss graph for ResNet 18 trained with non-normalized dataset.***

![16](https://github.gatech.edu/storage/user/54998/files/77970be2-6d10-4232-8358-56db0429a7bf)
![17](https://github.gatech.edu/storage/user/54998/files/8cf173e9-5a83-4d7e-a96e-6d20bb282eda)
![18](https://github.gatech.edu/storage/user/54998/files/0a33a8db-f214-43b5-b537-3fe85f3d2ed1)<br/>
***Figure 5. Model accuracy, average distance, and loss graph for ResNet 18 trained with normalized dataset.***

![19](https://github.gatech.edu/storage/user/54998/files/1b13139e-8c82-4d4c-bd2d-96a071e1c228)<br/>
***Figure 6. Confusion matrix of VGG 16 trained with non-normalized dataset. X-axis indicates the predicted labels. Y-axis indicates the true labels.***

![20](https://github.gatech.edu/storage/user/54998/files/590e4fba-f907-4b0e-a40b-979ae58bb0c5)<br/>
***Figure 7. Confusion matrix of VGG 16 trained with normalized dataset. X-axis indicates the predicted labels. Y-axis indicates the true labels.***

![21](https://github.gatech.edu/storage/user/54998/files/b01d05be-7902-4589-81a7-1d040364b886)<br/>
***Figure 8. Confusion matrix of ResNet 18 trained with non-normalized dataset. X-axis indicates the predicted labels. Y-axis indicates the true labels.***

![22](https://github.gatech.edu/storage/user/54998/files/e0d85e73-7dcd-4b72-8426-9e7947856a42)<br/>
***Figure 9. Confusion matrix of ResNet 18 trained with normalized dataset. X-axis indicates the predicted labels. Y-axis indicates the true labels.***

| First Header  | VGG16 | ResNet18  |
| :--- | :---: | :---: |
| Non-normalized  | 33.2%  | 29%  |
| Normalized  | 25%  | 29%  |

***Table 1. Test accuracy for the four experiments.***

## References
U.S. Department of Health and Human Services. (n.d.). Diabetic retinopathy data and statistics. National Eye Institute. Retrieved October 2, 2021, from https://www.nei.nih.gov/learn-about-eye-health/outreach-campaigns-and-resources/eye-health-data-and-statistics/diabetic-retinopathy-data-and-statistics. 

Mayo Foundation for Medical Education and Research. (2021). Diabetic retinopathy. Mayo Clinic. Retrieved October 2, 2021, from https://www.mayoclinic.org/diseases-conditions/diabetic-retinopathy/symptoms-causes/syc-20371611. 

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

Gulshan, Varun, et al. "Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs." Jama 316.22 (2016): 2402-2410.
Simonyan, K., & Zisserman, A. (2014). Very dee
