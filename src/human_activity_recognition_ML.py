#!/usr/bin/env python
# coding: utf-8

# Human Activity Recognition in Smarthomes
# Supervised machine learning modeling

# importing required libraries
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from xgboost.sklearn import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def load_data(path):
    ''' path should be string showing the directory of the clean dataset from EDA.py '''
    df = pd.read_csv(path)
    return df

path = input('Insert The directory where your dataset is located: ')
df = load_data(str(path))

def X_y(df):
    y = df['activity']
    X = df.drop('activity', axis = 1)
    return X, y

X, y = X_y(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

def scaling(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test)
    X_train = pd.DataFrame(X_train, columns = X.columns)
    X_test = pd.DataFrame(X_test, columns = X.columns)
    return X_train, X_test

def imputing(X_train, X_test):
    imp_freq = SimpleImputer(strategy='median')
    X_train = imp_freq.fit_transform(X_train)
    X_test = imp_freq.transform(X_test)
    X_train = pd.DataFrame(X_train, columns = X.columns)
    X_test = pd.DataFrame(X_test, columns = X.columns)
    return X_train, X_test

def undersampling(X_train, y_train):
    strategy = {}
    for i in y_train.value_counts().index[:3]:
        strategy[i] = 20000
    X_train_arr = np.array(X_train)
    rus = RandomUnderSampler(sampling_strategy=strategy, random_state=0)
    X_res, y_res = rus.fit_sample(X_train_arr, y_train)
    X_train = X_res
    y_train = y_res
    return X_train, y_train

y_train.value_counts() / len(y_train)

def oversampling(X_train, y_train):
    X_train_arr = np.array(X_res)
    os_smote = SMOTE(sampling_strategy='not majority')
    X_res, y_res = os_smote.fit_sample(X_train_arr, y_res)
    X_train = X_res
    y_train = y_res
    return X_train, y_train

lrc = LogisticRegression(multi_class='ovr')
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier()
abc = AdaBoostClassifier()
xgb = XGBClassifier()                  
mlp = MLPClassifier(tol=1e-3, random_state=0)

models = [lrc, dtc, rfc, gbc, abc, xgb, mlp]
names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 
             'Ada Boosting', 'XGBoosting', 'MLP Classifier']
 
def evaluation_metrics(lst):
    ''' lst is the list of your desired performance metrics)
    '''
    evaluation = pd.DataFrame(columns = lst)
    return evaluation

# The performance metric list that I have defined in my model is:
lst = ['accuracy', 'elapsed time[s]']

# building classifier
def classifier(X,y, model):
    time_start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, preds), 3)
    elapsed_time = round((time.time()-time_start), 2)
    return preds, accuracy, elapsed_time

def run_models():
    ''' calling this function takes almost 3 hours with 4CPUs '''
    for model, name in zip(models, names):
        classifier(X, y, model)
        evaluation.loc[names, 'accuracy'] = preds
        evaluation.loc[names,'elapsed time[s]'] = elapsed_time
    
def feature_imp_plot(model, X):
    ''' plots the feature importance '''
    feature_importances = 100*model.feature_importances_ / np.sum(model.feature_importances_)
    feat_importances = pd.Series(feature_importances, index=X.columns)
    feat_importances.nlargest(20, keep='first')[::-1].plot(kind='barh')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Relative feature importances in Random Forest')
    plt.savefig('feature_importances_rfc.png', bbox_inches='tight')
    return

def plot_cm(model, X_test, y_test):
    ''' plots the confusion matrix '''
    fig, ax = plt.subplots(figsize=(20, 20))
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax);
    return

def plot_cm_percent(model, X_test, y_test):
    ''' plots the normalized confusion matrix '''
    fig, ax = plt.subplots(figsize=(20, 20))
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax, normalize='true');
    return

# Model selection
def cross_val(X, y, model):
    ''' returns predictions and predictions proba for CV '''
    kf = KFold(n_splits=4, shuffle=True, random_state=0)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        X_test.shape
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train,y_train)
        preds=model.predict(X_test)
        proba=model.predict_proba(X_test)
    return preds, proba

def model_tuning(X, y, model, param):
    ''' returns the best parameter for hyperparameter tuning step '''
    scorings = ['accuracy']
    GCV = GridSearchCV(model, param_grid=param, scoring=scorings, verbose=1, refit='accuracy')
    GCV.fit(X_train, y_train)
    GCV.cv_results_
    return GCV.best_score_, GCV.best.params_

# I used below params for random forest hyperparameter tuning:
param = {'n_estimators': [20, 50, 100, 200],
     'criterion': ['gini', 'entropy'],
     'max_depth': [None, 5, 6, 7]}
    
# feature reduction
def PCA_features(X, y, n):
    '''n is the number of component'''
    start = time.time()
    pca = PCA(n_components=n)
    pca_X = pca.fit_transform(X)
    print('Shape of pca_reduced data is: ', pca_X.shape)
    pca_X = np.vstack((pca_X.T, y)).T
    pca_df = pd.DataFrame(pca_X)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    if n==2:
        pca_df = pd.DataFrame(pca_X, columns=['1st_principal', '2nd_principal', 'activity'])
        sns.FacetGrid(pca_df, hue='activity', size=8).map(plt.scatter, '1st_principal', '2nd_principal').add_legend();
    print('PCA done! Time elapsed: {} seconds'.format(time.time()-start))
    return pca_df

def plot_pca(X):
    ''' plots the cumulative explained variance vs number of PCA components '''
    pca = PCA(n_components=X.shape[1])
    pca_X = pca.fit_transform(X)
    percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
    cum_var_explained = np.cumsum(percentage_var_explained)
    # plotting the PCA spectrum
    plt.clf()
    plt.figure(figsize=(10,6))
    plt.plot(cum_var_explained, linewidth=4);
    plt.axis('tight')
    plt.grid(False)
    plt.xlabel('Number of components', fontsize = 20)
    plt.ylabel('Cumulative explained variance', fontsize = 20)
    plt.title('Dimensionality reduction - PCA', fontsize = 20)
    return

def tsne_features(X, y, n):
    '''n is the number of component'''
    ''' calling this function with n=2 takes more than an hour with 4CPUs '''
    start = time.time()
    tnse = TSNE(n_components=n, random_state=0)
    tnse_X = tnse.fit_transform(X)
    tnse_X = np.vstack((tnse_X.T, y)).T
    tnse_df = pd.DataFrame(tnse_X, columns=['1st_dimension', '2nd_dimension', 'activity'])
    # if n==2:
    #     sns.FacetGrid(tnse_df, hue='activity', size=8).map(plt.scatter, '1st_dimension', '2nd_dimension').add_legend();
    # print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    return tnse_df