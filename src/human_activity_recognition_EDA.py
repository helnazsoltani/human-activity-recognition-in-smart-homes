#!/usr/bin/env python
# coding: utf-8

# Human Activity Recognition in Smarthomes
# EDA and Data Visualization

# importing required libraries
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import glob
import time
# import datetime
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits import mplot3d
from sklearn.preprocessing import LabelEncoder

start = time.time()

def load_data(path):
    ''' path should be of string type showing the directory of the dataset '''
    df = pd.read_csv(path)
    return df

def load_data_all(path):
    '''path should include /*csv to load them all together'''
    files = glob.glob(path)
    all_df = [pd.read_csv(element) for element in files]
    df = all_df[1]
    for i in range(len(all_df)):
        if (df.columns != all_df[i].columns).sum() == 0:
            print(i)
            frames = [df, all_df[i]]
            df = pd.concat(frames)
        else:
            print('The features do not match in these two dataframe.')
    return df

path = 'C:/Users/dell/data/casas-dataset/csh101/csh101.ann.features.csv'

# path = input('Insert The directory where your dataset is located: ')
df = load_data(str(path))
      
def df_specs(df):
    ''' returns the high-level specs of the dataframe '''
    print('There exists {} observations and {} features in train dataset. \n'.format(df.shape[0], df.shape[1]))
    print('No. of null values in daatset: {}'.format(df.isnull().sum().sum()))
    if df.isnull().sum().sum() != 0:
        print('There are some null values in the dataframe, run df.isnull().sum().sum()) for more detail.')
        answer = input('Do yo wish to drop null values? Y/N?: ')
        if answer == 'Y' or answer == 'y':
            ''' if there are some null values and you would like to remove them, you can run this function,
                or else you can use imputing function in ML code if you do not wish to drop nulls
            '''
            df = df.dropna()      
        else:
            print('I do not want to drop null values')
    print('No. of duplicates in dataset: {}'.format(sum(df.duplicated())))
    if df.duplicated().sum() != 0:
        df.drop_duplicates(inplace=True)
    print('No. of classes in target variable: {}'.format(df['activity'].nunique()))
    return df
        
# to see the first 5 observations
# df.head()

# to check for datatype
# df.info()

# to check for statistics of numerical features
# df.describe()

def num_datapoints_class(df):
    ''' returns the number of observations in each classes of target variable '''
    return df['activity'].value_counts()

def dropped_cols(df):
    ''' drops the columns with constant value '''
    k = 0
    for i in df.columns[:-1]:
        if df[i].std() == 0:
            k += 1
            df.drop(i, axis=1, inplace=True)
            print('Column {} was dropped'.format(i))   
    if k == 0:
        print('No column {} was dropped'.format(i))  
    return

def last_event(df):
    ''' checks for multicollinearity of lastSensorEventSeconds and lastSensorEventHours '''
    df['hour'] = df['lastSensorEventSeconds'] // 3600 
    # df['lastSensorEventMinutes'] = (df['lastSensorEventSeconds'] // 60) %60
    if (df['hour'] == df['lastSensorEventHours']).mean() == 1:
        df.drop('lastSensorEventHours', axis=1, inplace=True)
        print('Column lastSensorEventHours has been removed.')
    return

def class_balance(df):
    ''' check the balance of dataframe '''
    percentage = df['activity'].value_counts() / len(df['activity'])
    if (percentage.max() - percentage.min())> 0.2:
        print('The data is pretty imbalanced and needs to be taken care of.')
    return

def sensor_data(df):
    ''' checks if all 11 sensors have data in the dataset '''
    for i in ['lastMotionLocation', 'lastSensorLocation', 'lastSensorID']:
        if df[i].nunique() != 11:
            print('Not all the sensors are in the list of {}'.format(i))
    if (df['lastSensorID'] != df['lastSensorLocation']).sum() == 0:
        df.drop('lastSensorLocation', axis=1, inplace=True)
        print('\'lastSensorLocation\' has been removed.')
    else:
        print('\'lastSensorLocation\' and \'lastSensorLocation\' do not share the same info.')
    return df

def sensor_check(df):
    ''' returns the features corresponding to all sensors by filtering the columns' name '''
    sensorCount = df.filter(like='sensorCount').columns
    sensorElTime = df.filter(like='sensorElTime').columns
    # cols_hist = sensorCount.append(sensorElTime)
    return sensorCount, sensorElTime

def onehot_encoding(df, column_list):
    df = pd.get_dummies(df, columns=column_list)
    return df

# Here is the list of columns I did one-hot encoding:
# cat_cols = ['lastSensorDayOfWeek', 'prevDominantSensor1', 'prevDominantSensor2', 'lastSensorID', 'lastMotionLocation']

def save_clean(pathname_csv):
    ''' saves the clean dataset for future use '''
    df.to_csv(pathname_csv)
    return

def feature_distribution(df, feature):
    ''' plots the distribution of categorical features, feature should be input of string type '''
    percentage = df[feature].value_counts() / df.shape[0]*100
    plt.figure(figsize=(12, 12))
    sns.countplot(data=df, x=feature, order=percentage.index, color='b')
    plt.xlabel('')
    plt.xticks(rotation=30, horizontalalignment='right')
    plt.ylabel('counts')
    # plt.ylim(0, 2000)
    plt.title('Activity counts');
    for i in range(percentage.shape[0]):
        strt = '{:0.1f}%'.format(percentage[i])
        plt.text(i, df['activity'].value_counts()[i] + 1000, strt, ha = 'center', size = 14)
    # plt.savefig('../figure/activity_counts', bbox_inches = 'tight');
    return

# feature_distribution(df, 'activity')

# Let's perform some EDA on the first n activities (with highest distribution in dataset)
def sub_df(df, number):
    ''' creates subsets of dataframe for i'th (number input) activities with highest majority'''
    df_name = df['activity'].value_counts().index[number]
    return df_name, df[df['activity'] == df['activity'].value_counts().index[number]]

def count_plot(feature, n):
    ''' plots the feature distibution per individual activity,
    n is the i'th class (with highest majority) you wish to investigate more'''
    plt.figure(figsize=(12, 12))
    for i in range(n):
        df_name, df_sub = sub_df(df, i)
        sns.distplot(df_sub[feature], hist=False, label=df_name)
        plt.xlabel(feature);
        plt.grid(False)
    # plt.savefig('../normalized_tBodyAccMagmean', bbox_inches = 'tight');
    return

# count_plot('lastSensorDayOfWeek', 15)

def box_plot(feature):
    ''' box-plots the feature statistics per individual activities '''
    plt.figure(figsize=(12, 12))
    sns.boxplot(data=df, x='activity', y=feature, showfliers=False)
    plt.xticks(rotation=30, horizontalalignment='right')
    return

# box_plot('sensorCount-Kitchen')

def hist_plot(df):
    ''' hist-plots the dataframe '''
    df.hist(grid=False, figsize=(10, 10));
    plt.tight_layout()
    
# hist_plot(df['sensorCount-Kitchen'])
  
def scatter_plot(df, column1, column2):
    '''' scatter-plots two columns per different activity '''
    label_encoder = LabelEncoder() # Encode target labels with value between 0 and n_classes-1
    y = label_encoder.fit_transform(df['activity']) # labeling the target variables
    plt.scatter(df[column1], df[column2], c=y, alpha=0.2, marker='o', s=30);
    return 
    
# scatter_plot(df, 'sensorCount-Kitchen', 'sensorElTime-Kitchen')

def scatter_plot_3d(df, column1, column2):
    ''' 3D version of scatter_plot function defined above,
    3D-scatter plots of two columns per different activity '''
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['activity']) # labeling the target variables
    ax = plt.axes(projection='3d')
    ax.scatter3D(df[column1], df[column2], y, c=y);
    return ax

# scatter_plot_3d(df, 'sensorCount-Kitchen', 'sensorElTime-Kitchen')

print('Time elapsed: {} seconds'.format(round(time.time()-start),3))