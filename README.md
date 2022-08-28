# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
``` python3
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("data.csv")

df.tail()

x=df[["Input"]].values

y=df[["Output"]].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

import tensorflow as tf

model=tf.keras.Sequential([tf.keras.layers.Dense(16,activation='relu'),
                           tf.keras.layers.Dense(8,activation='relu'),
                           tf.keras.layers.Dense(1)])
model.compile(loss="mae",optimizer="adam")

history=model.fit(x_train,y_train,epochs=2000)

import numpy as np

pred=model.predict(x_test)
pred

tf.round(model.predict([[20]]))

pd.DataFrame(history.history).plot()

r=tf.keras.metrics.RootMeanSquaredError()
r(y_test,pred)
```

## Dataset Information

![image](https://user-images.githubusercontent.com/75235022/187082580-a63beb78-af92-4e0a-8ea5-e9a7f5a8f952.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75235022/187082567-8c8ebc24-ca23-4b9f-ae6d-293b29fb88fd.png)

### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/75235022/187082598-3322ac46-f313-4d78-bab1-4e0c74ce50e1.png)

### New Sample Data Prediction
![image](https://user-images.githubusercontent.com/75235022/187082616-f961f9d5-e459-4988-9102-c87d313648db.png)

## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully
