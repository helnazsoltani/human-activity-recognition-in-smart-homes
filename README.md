# Human Activity Recognition in Smart Homes
## by Helnaz Soltani
### June 08, 2020

![image](https://s27389.pcdn.co/wp-content/uploads/2017/07/AdobeStock_109710668-1024x420.jpeg)


## Table of Contents
- [Introduction](#Introduction)
    - [Motivation](#Motivation)
    - [Objectives](#Objectives)
    - [Applications](#Applications)
- [Dataset](#Dataset)
- [Data Wrangling](#Data-Wrangling)
    - [Data Preparation](#Data-Preparation)
    - [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Machine Learning Modeling](#Machine-Learning-Modeling)
    - [Model Selection](#Model-Selection)
    - [Performance Metrics](#Performance-Metrics)
    - [Model Tuning](#Model-Tuning)
    - [Resampling Data](#Resampling-Data)
    - [Dimensionality Reduction](#Dimensionality-Reduction)
    - [Generalizing Developed Models to all Datasets](#Generalizing-Developed-Models-to-all-Datasets)
- [Conclusions](#Conclusions)
    - [Concluding Remarks](#Concluding-Remarks)
    - [Challenges](#Challenges)
    - [Future Plans](#Future-Plans)
- [Project Files](#Project-Files)
    - [Deliverables](#Deliverables)
    - [System Requirements](#System-Requirements)
- [Acknowledgements](#Acknowledgements)

## Introduction
### Motivation
Human Activity Recognition (HAR) got a tremendous amount of attention due to the growth of technology. The required data is collected by sensors which can be either environmental (e.g. smart home) or wearable (e.g. smart device). Therefore, it is of great interest to build a comprehensive HAR methodology to predict human activity from sequential sensor data.

### Objectives
The classification task is to map a sequence of environmental sensor events in a smart home to a corresponding activity of its residents.

### Applications
HAR have many applications, which some of the important ones can be listed as:
- Health care
- Energy saving
- Security and safety

<a href="#Human-Activity-Recognition-in-Smart-Homes">Back to top</a>

## Dataset
- The datasets and smart homes' layout can be found in [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+from+Continuous+Ambient+Sensor+Data). However, I put a sample of 50k observations from one of them in the [data](https://github.com/helnazsoltani/human-activity-recognition-in-smart-homes/tree/master/data) folder (I kept the file name according to the original file for the sake of consistency).
- The datasets reflect the info from motion sensors, door sensors, and light sensors in 30 smart homes which have recorded the activity of the volunteer residents.
- There exist 11 sensors in each dataset, each of them corresponding to a location in smart home.
- Most of the sensor data was collected continuously for two months while residents performed their normal routines, though some of them contain labels for extended time periods.
- The raw data was processed in the form of feature vectors using a sliding window of 30 sensor events.
- Combining all 30 smart homes, the total number of observations are 7M+ with 45 classes of activity.
- The 1st smart home which I considered for EDA and initial modeling has 320k+ observations and 35 classes of activity, then I generalized the model to the 30-smart-home dataset.

<a href="#Human-Activity-Recognition-in-Smart-Homes">Back to top</a>

## Data Wrangling

### Data Preparation
The below steps have been taken in order to prepare the data:
- Data cleaning issues
    - Duplicated entries
    - Null values
    - Incorrect data type
    - Low variance statistics
- Data imbalance
- One-hot encoding for categorical feature
- Saving the cleaned dataframe in a csv file

### Exploratory Data Analysis
These findings represent some of the selected EDAs:
- Number of observations per activity:
    - A great majority of activities belong to 'Other Activity'.
    - The rest of activities (which were not shown in the below plot) have less than 1% distribution.

![image](https://github.com/helnazsoltani/human-activity-recognition-in-smart-homes/blob/master/figures/activity_distribution.png)

- Last sensor event distribution: The sensors record less activity during the normal sleeping time.

![image](https://github.com/helnazsoltani/human-activity-recognition-in-smart-homes/blob/master/figures/last_sensor_event_seconds.png)

- Window duration (time duration of the 30 event sliding window) for the first 15 activities with highest frequency: The activities with the highest majority seem to have smaller window duration.

![image](https://github.com/helnazsoltani/human-activity-recognition-in-smart-homes/blob/master/figures/window_duration_seconds_15.png)

- Identifying multicollinear features such as 'lastSensorEventHours' and 'lastSensorEventSeconds'
- Scatterplot 'sensorElTime-sensorname' and 'sensorCount-sensorname' for all sensors
- 3d-plot 'sensorElTime-sensorname', 'sensorCount-sensorname', and 'activity'

<a href="#Human-Activity-Recognition-in-Smart-Homes">Back to top</a>

## Machine Learning Modeling
### Model Selection
I developed the below models for one dataset:
- Logistic regression
- Decision tree
- Random forest
- Gradient boosting
- XGBoosting
- Ada boosting
- Multi-layer perceptron

### Performance Metrics
- Accuracy metric to evaluate how the model predicts the activity:
    - Among all of the above models, Random Forest, XGBoosting, and Decision Tree performed the best.
    
- Confusion matrix to capture which activities were mislabeled in my model:
    - Reviewing the confusion matrix of the Random Forest, the majority of the misclassified predictions are related to the non-defined class (Other Activity).
    
- Elapsed run time
    - Comparing the elapsed run time of the 3 above-mentioned models, I decided to pursue further study on Random Forest. 

![image](https://github.com/helnazsoltani/human-activity-recognition-in-smart-homes/blob/master/figures/models_accuracy_runtime.png)


#### Feature importance plot for Random Forest

![image](https://github.com/helnazsoltani/human-activity-recognition-in-smart-homes/blob/master/figures/feature_importances_rfc.png)

### Model Tuning
- I performed hyperparameter tuning for Random Forest on one dataset, however, the default values do the great job.

### Resampling Data
- I applied undersampling for 'Other_Activity' category (which has the majority in the dataset) and oversampling for the rest of categories. It turned out the model performs better on the original imbalanced data.

### Dimensionality Reduction
I applied PCA and tSNE for dimensionality reduction purposes. 
- For PCA, with 7 numbers of components, maximum cumulative explained variance is achieved.

![image](https://github.com/helnazsoltani/human-activity-recognition-in-smart-homes/blob/master/figures/pca_n_components.png)

- For tSNE with 2 numbers of components, the activities are not separated.

### Generalizing Developed Models to all Datasets
- Due to memory issues, I was only able to generalize the developed Logistic Regression and Decision Tree models to the combined dataset (consists of 30 smart homes). The accuracy of Decision Tree is 92.3% and the run time is 294 sec.

<a href="#Human-Activity-Recognition-in-Smart-Homes">Back to top</a>

## Conclusions
### Concluding Remarks
- The predictive model is able to recognize human activity in a smart home with 97% accuracy.
- The prediction error is for the categories that have not been defined clearly.
- The data was processed fairly well by domain experts.

### Challenges
- I planned to compare all the models on the combined 30-smart-home dataset for generalization purposes. However, I could not increase my AWS memory to run Spark, therefore, I was able to check the performance for Logistic Regression and Decision Tree models only.
- There are some activities that have not been predefined, the model may be more predictive if it gets learnt in an unsupervised manner. 

### Future Plans
- To generalize and improve the model to the 30-smart-home dataset which makes my predictive model independent of floorplan
- To develop a neural network model
- To approach the problem unsupervisingly
- To generalize the model to wearable sensors such as smart device and smart watch

<a href="#Human-Activity-Recognition-in-Smart-Homes">Back to top</a>

## Project Files
### Deliverables
- [EDA python code](src/human_activity_recognition_EDA.py)
- [Machine learning modeling python code](src/human_activity_recognition_ML.py)
- [Presentation](to be uploaded)

### System Requirements
- Python 3.7.3
- Required libraries for running EDA code: numpy, pandas, glob, time, datetime, matplotlib, seaborn, mpl_toolkits
- Required libraries for running Machine Learning code: pandas, time, matplotlib, seaborn, sklearn, xgboost, imblearn
- All codes have been running on AWS Sagemaker with 4 CPUs.

<a href="#Human-Activity-Recognition-in-Smart-Homes">Back to top</a>

### Acknowledgements
This work is completed as a part of fulfillment for the [Galvanize DSI](https://github.com/GalvanizeDataScience) degree.
I would like to thank [Hamid](https://github.com/Hamidmol), [Tomas](https://github.com/tomasbielskis), and the entire Galvanize team for their support and guidance during the course of study.

<a href="#Human-Activity-Recognition-in-Smart-Homes">Back to top</a>
