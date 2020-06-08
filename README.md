# Human Activity Recognition in Smarthomes
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
- [Predictive Machine Learning Modeling](#Predictive-Machine-Learning-Modeling)
    - [Model Selection](#Model-Selection)
    - [Performance Metrics](#Performance-Metrics)
    - [Model Tuning](#Model-Tuning)
    - [Resampling Data](#Resampling-Data)
    - [Dimensionality Reduction](#Dimensionality-Reduction)
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
Human Activity Recognition (HAR) got tremendous amount of attentions due to the growth of technology. The required data is collected by sensors which can be either environmental (e.g. smart home) or wearable (e.g. smart device). Therefore, it is of great interest to build a comprehensive HAR methodology to predict human activity from sequential sensor data.

### Objectives
The classification task is to map a sequence of environmental sensor events in a smart home to a corresponding activity of its residents.

### Applications
HAR have many applications, which some of the important ones can be listed as:
- Health care
- Energy saving
- Security and safety

<a href="#Human-Activity-Recognition-in-Smarthomes">Back to top</a>

## Dataset
- The datasets and smart homes' layout can be found in [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+from+Continuous+Ambient+Sensor+Data). However, I put a sample of 50k observations from one of them in the [data](https://github.com/helnazsoltani/human-activity-recognition-in-smarthomes/tree/master/data) folder (I kept the file name according to the original file for the sake of consistency).
- The datasets reflect the info from motion sensors, door sensors, and light sensors in 30 smarthomes which have recorded the activity of the volunteer residents.
- There exist 11 sensors in each dataset, each of them is corresponding to a location in smarthome.
- Most of the sensor data were collected continuously for two months while residents performed their normal routines, though some of the them contain labels for extended time periods.
- The raw data was processed in the form of feature vectors which was generated using a sliding window of 30 sensor events. 

<a href="#Human-Activity-Recognition-in-Smarthomes">Back to top</a>

## Data Wrangling
### Data Preparation
- Data cleaning
    - Duplicated entries
    - Null values
    - Incorrect data type
    - Low variance statistics
- Data imbalance
- One-hot encoding for categorical feature
- Saving the cleaned dataframe in a csv file

### Exploratory Data Analysis
- Number of observations per activity: A great majority of activities belong to 'Other Activity'

![image](https://github.com/helnazsoltani/human-activity-recognition-in-smarthomes/blob/master/figures/activity_distribution.png)

- Last sensor event distibution: The sensors record less activity during the normal sleeping time.

![image](https://github.com/helnazsoltani/human-activity-recognition-in-smarthomes/blob/master/figures/last_sensor_event_seconds.png)

- Window duration for the first 15 activities with highest frequency

![image](https://github.com/helnazsoltani/human-activity-recognition-in-smarthomes/blob/master/figures/window_duration_seconds_15.png)

- Identifying multicollinear features
- Scatterplot 'sensorElTime-sensorname' and 'sensorCount-sensorname' for all sensors
- 3d-plot 'sensorElTime-sensorname', 'sensorCount-sensorname', and 'activity'.

<a href="#Human-Activity-Recognition-in-Smarthomes">Back to top</a>

## Predictive Machine Learning Modeling
### Model Selection
I developed the below models to one dataset:
- Logistic regression
- Decision tree
- Random forest
- Gradient boosting
- XGBoosting
- Ada boosting
- Multi-layer perceptron

### Performance Metrics
- Accuracy metric to evaluate how the model predicts the activity
- Confusion matrix to capture which activities were mislabeled in my model
- Elapsed run time

![image](https://github.com/helnazsoltani/human-activity-recognition-in-smarthomes/blob/master/figures/models_accuracy_runtime.png)

Among all of the above models, Random Forest, XGBoosting, and Decision Tree performed the best.
Comparing the elapsed run time of the 3 above-mentioned models, I decided to pursue further study on Random Forest as it runs way faster compared to the other two.

#### Feature importance plot for Random Forest

![image](https://github.com/helnazsoltani/human-activity-recognition-in-smarthomes/blob/master/figures/feature_importances_rfc.png)

### Model Tuning
- I performed hyperparameter tuning for Random Forest on one dataset, however, the default values do the great job.

### Resampling Data
- I applied undersampling for 'Other_Activity' category (which has the majority in the dataset) and oversampling for the rest of categories. It turned out the model performs better on the original imbalanced data.

### Dimensionality Reduction
I applied PCA and tSNE for dimensionality reduction purposes. 
- For PCA, with 7 numbers of components, maximum cumulative explained variance is achieved.

![image](https://github.com/helnazsoltani/human-activity-recognition-in-smarthomes/blob/master/figures/pca_n_components.png)

- For tSNE with 2 numbers of components, the activities are not separated.

<a href="#Human-Activity-Recognition-in-Smarthomes">Back to top</a>

## Conclusions
### Concluding Remarks
- The predictive model is able to recognize human activity in a smarthome with 97% accuracy.
- The prediction error is for the categories that have not been defined clearly.
- The data was processed fairly well by domain experts.

### Challenges
- I planned to compare all the models on the combined 30-smarthome dataset for generalization purposes. However, I could not increase my AWS memory to run Spark, therefore, I was able to check the performance for Logistic Regression and Decision Tree models only.
- There are some activities that have not been predefined, the model may be more predictive if it gets learnt in an unsupervised manner. 

### Future Plans
- To generalize and improve the model to the 30-smarthome dataset which makes my predictive model independent of floorplan
- To develop a neural network model
- To approach the problem unsupervisingly
- To generalize the model to wearable sensors such as smart device and smart watch

<a href="#Human-Activity-Recognition-in-Smarthomes">Back to top</a>

## Project Files
### Deliverables
- [EDA python code](src/human_activity_recognition_EDA.py)
- [Machine learning modeling python code](src/human_activity_recognition_ML.py)
- [Presentation](to be uploaded)

### System Requirements
- Python 3.7.3
- Required libraries for running EDA code: numpy, pandas, glob, time, datetime, matplotlib, seaborn, mpl_toolkits
- Required libraries for running Machine Learning code: pandas, time, matplotlib, seaborn ,sklearn, xgboost, imblearn
- All codes have been running on AWS Sagemaker with 4CPUs.

<a href="#Human-Activity-Recognition-in-Smarthomes">Back to top</a>

### Acknowledgements
This work is completed as a part of fulfillment for the [Galvanize DSI](https://github.com/GalvanizeDataScience) degree.
I would like to thank [Hamid](https://github.com/Hamidmol), [Tomas](https://github.com/tomasbielskis), and the entire Galvanize team for their support and guidance during the course of study.

<a href="#Human-Activity-Recognition-in-Smarthomes">Back to top</a>
