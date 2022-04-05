# import library
import pandas as pd
import numpy as np


# read data file
heartdisease = pd.read_csv('data\cleaned_dataset.csv')
df = heartdisease.copy()


# convert multi-class 'Race' column
dummy = pd.get_dummies(df['Race'], drop_first=True)
df = pd.concat([df,dummy],axis=1)
del df['Race']


# Separating X and y
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']


# imort model packages
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler


# instantiate under sampler, fit it to the data, then resample the data
X_us, y_us = RandomUnderSampler(sampling_strategy="majority").fit_resample(X, y)

# build up pipeline using optimal model
estimators = [
    ('scale', StandardScaler()),
    ('logistic', LogisticRegression(max_iter=200)) 
]

# instantiate pipeline
pipe = Pipeline(estimators)

# fit the optimal model
pipe.fit(X_us, y_us)

# Saving model into pkl file
import joblib
joblib.dump(pipe, 'optimal_model.pkl')

