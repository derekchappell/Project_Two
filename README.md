# Project_Two has a surface level simplicity to the quesiton we would be asking of our data, can you predict home prices with the various physical attributes of a home? # 

-In this project we would be bring together our API skillsets, our Python knowledge and bringing it all together with machine learning technniques to see if we can answer the above stated question

## Loading all necessary libraries into our work book ##

```python
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
```

[.]()

## Normalizing the data will require us to use get dummies for both construction year and zip code ##

```python
rf_df = ma_df[['SitusZip','LandSize', 'LivableSqFootage', 'ConstructionYear', 'Pool', 'FullCashValue']].copy()
rf_df = pd.get_dummies(rf_df, columns=["SitusZip", 'ConstructionYear'])
rf_df = rf_df[rf_df.FullCashValue != 0]
```

## We need to establish a Y value or target for the model to attempt to predict, as stated in the introduction this is the Full Cash Value of a home ##
 - First we drop the column from our dataset then we establish it as our target value
```python

X = rf_df.copy()
X.drop("FullCashValue", axis=1, inplace=True)

y = rf_df["FullCashValue"].values.reshape(-1, 1)
y[:5]
```

## Here we are training the model based off of our splitting of the data covered above ##
 -  these lines of code can be exceuted in a block as shown below however it is best to break out certain sectors of code to aid in trouble shooting if an error is thrown
 
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_fit = rf_model.fit(X_train_scaled, y_train.ravel())
predictions = rf_fit.predict(X_test_scaled)
```

## Lets see what are the most important attributes of our data in determing value, sq footage and lot size were factors we had predicted would be most important  ##

```python
[(0.4647436496635477, 'LivableSqFootage'),
 (0.2464154409298598, 'LandSize'),
 (0.033440283386282375, 'ConstructionYear_2002'),
 (0.03322693374371636, 'SitusZip_85032.0'),
 (0.01982897673119421, 'SitusZip_85006.0'),
 (0.0148564424240925, 'SitusZip_85016.0'),
 (0.009557882758952273, 'SitusZip_85040.0'),
 (0.00904575126139606, 'ConstructionYear_2019'),
 (0.008255666873345231, 'SitusZip_85008.0'),
 (0.007005156176119154, 'SitusZip_85035.0')]
```
### Sure enough our predictions are correct ###

## We want a high R^2 and as low as possible a score for our RMSE score ##

```python
R^2 score:                 0.77
RMSE score:       502,073,729.14
```
### This is an acceptbale starting point for our model and shows some promise for further testing and tweaking, however it is quite impressive what these libraries are capable of right out of the box ###


## What did our linear regression model manage to achieve? ##
 - Spoiler Alert, it was not as impressive but lets look at some of the cleaning techniques that were necessary for this model PRIOR to running as well as some fun visualizations. Again it is impressive what these libraries are capable of out of the box but as our insturctor says "crap in, crap out". 

## First lets check for outliers ##

```python
sns.boxplot(x=ma_df['FullCashValue'])
```

[.]()

## Lets take the easiest steps first then go further if necessary ##

```python
ma_df = ma_df[ma_df.FullCashValue != 0]
sns.boxplot(x=ma_df['FullCashValue'])
``` 
[.]()
 - Better
 
## Ok lets get a bit fancier with our cleaning, there are a miriade of ways to accomplish this however this one is easiest to understand at first viewing ##

```python
cols = ['FullCashValue']

Q1 = ma_df[cols].quantile(0.25)
Q3 = ma_df[cols].quantile(0.75)
IQR = Q3 - Q1

ma_df = ma_df[~((ma_df[cols] < (Q1 - 1.5 * IQR)) |(ma_df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

sns.boxplot(x=ma_df['FullCashValue'])
```

[.]()
 - Not too bad at all, we have decent sized normalized dataset that I believe we can work with 

## Lets check for trends ##

```python
sns.pairplot(ma_df[["LivableSqFootage", "ConstructionYear", "LandSize", 'FullCashValue' ]], diag_kind="kde")
```

[.]()
 - There is something looking like a trend in the top right there with our target and expected primary attribute (let's imagine we didn't have a list above showing this)

## Splitting our X & Y, Normalizing the Data, and splitting it up into train and testing versions ##

```python
X = ma_df[['LivableSqFootage', 'ConstructionYear', 'LandSize', 'Pool']]
y = ma_df['FullCashValue']

X_normalized = preprocessing.normalize(X, norm='l2')

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, random_state=42)
```

## Running our model ##

```python
regressor = LinearRegression(normalize=True)

regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)
print(f'R^2 score: {r2_score(y_true=y_test, y_pred=predictions):20,.2f}')
print(f'RMSE score: {mean_squared_error(y_true=y_test, y_pred=predictions, squared=True):20,.2f}'
```
### What were the scores? ###

 ```python
R^2 score:                 0.63
RMSE score:       749,345,484.10
```

### So as we can see the linear regression model is not as good given our dataset and attempts at predicting price. Anyone who has worked with SKLearn already has a pretty good idea that Random Forest will be hard to beat in most situations. Whats the next go to model? Probably a further itteration of Random Forest if I were to venture a guess. ###
