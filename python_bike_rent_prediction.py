
# This project focus on the bike share program’s rebalancing issue, aiming to answer a question: “How many bikes will meet users’ demand in a future certain time.”


# load required libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir('D:/Bike') # set working directory


# load train data
train = pd.read_csv("day.csv")


# head of train data
train.head(2)


# check variable type and data information
train.info()


# drop non useful columns 'instant' & 'dteday'
train = train.drop(['instant', 'dteday'], axis = 1)
print("Variables Name:",train.columns)


# rename some columns for understanding
train = train.rename(columns = {'yr': 'year',
                               'mnth': 'month',
                               'weathersit': 'weather',
                               'hum': 'humidity',
                               'cnt': 'count'})
print("Variables Name:",train.columns)


# list of numerical columns & categorical columns
num_col = ['temp', 'atemp', 'humidity', 'windspeed', 'casual','registered', 'count']
cat_col = ['season', 'year', 'month', 'holiday', 'weekday', 'workingday','weather']


# statistical summary of numerical columns
round(train[num_col].describe(),4)


# unique values in categorical columns
train[cat_col].nunique()


# count of unique categories in each columns
for c in cat_col:
    print("---- %s ---" % c)
    print(train[c].value_counts())


######################
#                    #
#   Missing Value    #
#                    #
######################

train.isnull().sum().sort_values(ascending = False)


# Visualization of Numerical Variables

###########################
#                         #
# Histogram of Numerical  #
#       Variables         #
#                         #
###########################

fig, axi = plt.subplots(ncols = 4, figsize = (20,5))

fig.suptitle("Histogram of Numerical Variables")
axi[0].hist(x = 'temp', data = train, edgecolor = 'black', color = 'tab:olive')
axi[0].set(xlabel = 'Temperature')
axi[1].hist(x = 'atemp', data = train, edgecolor = 'black', color = 'tab:olive')
axi[1].set(xlabel = 'ATemperature')
axi[2].hist(x = 'humidity', data = train, edgecolor = 'black', color = 'tab:olive')
axi[2].set(xlabel = 'Humidity')
axi[3].hist(x = 'windspeed', data = train, edgecolor = 'black', color = 'tab:olive')
axi[3].set(xlabel = 'WindSpeed')


# Skewness of Independent Numerical Variables
print("Skewness of Temperature:",round(train['temp'].skew(),2))
print("Skewness of ATemperature:",round(train['atemp'].skew(),2))
print("Skewness of Humidity:",round(train['humidity'].skew(),2))
print("Skewness of Windspeed:",round(train['windspeed'].skew(),2))


###########################
#                         #
#   Boxplor of Numerical  #
#       Variables         #
#                         #
###########################

fig, axi = plt.subplots(ncols = 4, figsize = (20,5))

fig.suptitle("Boxplot of Numerical Variables")
sns.boxplot(x = 'temp', data = train, orient = 'v', color = "tab:olive", ax = axi[0])
axi[0].set(xlabel = 'Temperature')
sns.boxplot(x = 'atemp', data = train, orient = 'v', color = "tab:olive", ax = axi[1])
axi[1].set(xlabel = 'ATemperature')
sns.boxplot(x = 'humidity', data = train, orient = 'v', color = "tab:olive", ax = axi[2])
axi[2].set(xlabel = 'Humidity')
sns.boxplot(x = 'windspeed', data = train, orient = 'v', color = "tab:olive", ax = axi[3])
axi[3].set(xlabel = 'Wind Speed')


##############################
#                            #
#  Scatter plot of Numerical #
#    Variables with Target   #
#                            #
##############################

fig, axi = plt.subplots(ncols = 4, figsize = (20,5))

fig.suptitle("Correlation plot of Numerical Variables with Target Variable")
sns.regplot(x = "temp", y = "count", data = train, color = "tab:olive", ax = axi[0])
axi[0].set(xlabel = "Temperature")
sns.regplot(x = "atemp", y = "count", data = train, color = "tab:olive", ax = axi[1])
axi[1].set(xlabel = "ATemperature")
sns.regplot(x = "humidity", y = "count", data = train, color = "tab:olive", ax = axi[2])
axi[2].set(xlabel = "Humidity")
sns.regplot(x = "windspeed", y = "count", data = train, color = "tab:olive", ax = axi[3])
axi[3].set(xlabel = "Wind Speed")


print('Correlation of Temperature with Target Variable:',round(train['temp'].corr(train['count']),2))
print('Correlation of ATemperature with Target Variable:',round(train['atemp'].corr(train['count']),2))
print('Correlation of Humidity with Target Variable:',round(train['humidity'].corr(train['count']),2))
print('Correlation of Windspeed with Target Variable:',round(train['windspeed'].corr(train['count']),2))


# Visualization of Categorical Variables
###########################
#                         #
# Barplot of Categorical  #
#       Variables         #
#                         #
###########################


# Mean Count of Year, Month & Weekday
fig, axi = plt.subplots(ncols = 3, figsize = (20,5))

fig.suptitle('Mean Count of Year, Month & Weekday')

year_mean = pd.DataFrame(train.groupby('year')['count'].mean()).reset_index()
month_mean = pd.DataFrame(train.groupby('month')['count'].mean()).reset_index()
weekday_mean = pd.DataFrame(train.groupby('weekday')['count'].mean()).reset_index()

sns.barplot(x = 'year', y = 'count', data = year_mean, color = 'tab:olive',ax = axi[0])
sns.barplot(x = 'month', y = 'count', data = month_mean,color = 'tab:olive', ax = axi[1])
sns.barplot(x = 'weekday', y = 'count', data = weekday_mean,color = 'tab:olive', ax = axi[2])


# mean count of Month & weekday with respect of Year

fig, axi = plt.subplots(ncols = 2, figsize = (20,5))
fig.suptitle('Count of Month & Weekday with respect of year')

sns.barplot(x = 'weekday', y = 'count', data = train, hue = 'year',ax = axi[0])
sns.barplot(x = 'month', y = 'count', data = train, hue = 'year',ax = axi[1])


# Mean count of Season & Weather

fig, axi = plt.subplots(ncols = 2, figsize = (20,4))

fig.suptitle('Mean Count of Season & Weather')
season_mean = pd.DataFrame(train.groupby('season')['count'].mean()).reset_index()
weather_mean = pd.DataFrame(train.groupby('weather')['count'].mean()).reset_index()

sns.barplot(x = 'season', y = 'count', data = season_mean, color = 'tab:olive',ax = axi[0])
sns.barplot(x = 'weather', y = 'count', data = weather_mean,color = 'tab:olive', ax = axi[1])

########################
#                      #
# Feature Enginerring  #
#                      #
########################

# create function for month_bin
def month_bin(df):
    if df['month'] <= 4 or df['month'] >= 11:
        return(0)
    else:
        return(1)


# create function for weekday_bin
def weekday_bin(df):
    if df['weekday'] < 2:
        return(0)
    else:
        return(1)

# create two new variables 'month_bin' & 'weekday_bin' and drop original variable 'month' & 'weekday'
train['month_bin'] = train.apply(lambda df: month_bin(df), axis = 1)
train = train.drop(['month'], axis = 1)
train['weekday_bin'] = train.apply(lambda df: weekday_bin(df), axis = 1)
train = train.drop(['weekday'], axis = 1)


###########################
#                         #
#   Heatmap of Numerical  #
#       Variables         #
#                         #
###########################

corr = train[num_col].corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots(figsize = (12,8))
sns.heatmap(corr, mask = mask, vmax = 7, square = True, annot = True)


##########################
#                        #
#   Chi Square Test of   #
#     Independence       #
#                        #
##########################

# list of Categorical Independent Variables
cat_col = ['season', 'year', 'holiday', 'workingday','weather','month_bin','weekday_bin']
category_paired = [(a,b) for a in cat_col for b in cat_col]

# make p-value matrix 
p_values = []
from scipy.stats import chi2_contingency
for f in category_paired:
    if f[0] != f[1]:
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(train[f[0]],train[f[1]]))
        
        p_values.append(p.round(4))
    else:
        p_values.append('NA')
p_values = np.array(p_values).reshape((7,7))
p_values = pd.DataFrame(p_values, index=cat_col, columns=cat_col)
print(p_values)


########################
#                      #
#  Feature Importance  #
#                      #
########################

# list of unwanted columns to check feature importance
unwanted_col = ['count', 'registered', 'casual']


# Check Feature importance with the help of Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_imp = RandomForestRegressor(n_estimators = 300)
x = train.drop(columns = unwanted_col)
y = train['count']
rf_imp.fit(x,y)
feature_imp = pd.DataFrame({'features': train.drop(columns = unwanted_col).columns,
                           'importance': rf_imp.feature_importances_})
feature_imp.sort_values(by = 'importance', ascending = False).reset_index(drop = True)



#########################
#                       #
#   Multi-Collinearity  #
#                       #
#########################

# Check VIF score of Independent Numerical Variables (before removal)
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF            
from statsmodels.tools.tools import add_constant
df_numeric = add_constant(train[['temp','atemp','humidity','windspeed']])
VIF = pd.Series([VIF(df_numeric.values, i) for i in range(df_numeric.shape[1])], 
                 index = df_numeric.columns)
print(VIF)


# Check VIF score of Independent Numerical Variables (after removal)
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF            
from statsmodels.tools.tools import add_constant
df_numeric = add_constant(train[['temp','humidity','windspeed']])
VIF = pd.Series([VIF(df_numeric.values, i) for i in range(df_numeric.shape[1])], 
                 index = df_numeric.columns)
print(VIF)


############################
#                          #
#  one hot enconding for   #
#     season & weather     #
#                          #
############################

# list of columns for dummy variables
one_hot_col = ['season','weather']


# define function to create dummy variables
def one_hot(df, cols):
    dummies = pd.get_dummies(df[cols], prefix=cols, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop([cols], axis = 1)
    return df

# create dummy variables
for cols in one_hot_col:
    train = one_hot(train, cols)


##############################
#                            #
#  removal of outliers from  #
#    humidity & windspeed    #
#                            #
##############################

# make copy of original dataframe
train_WO = train.copy()

# define function for outlier removal from selected columns
def remove_out(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1
    low_fen = q1-1.5*iqr
    upr_fen = q3+1.5*iqr
    df = df.loc[(df[col] > low_fen) & (df[col] < upr_fen)]
    return df

# list of outlier columns
outlier = ['humidity','windspeed']

# remove outlier form df 'train_WO'
print('Shape of Train Data before removing Outliers:', train_WO.shape)
for col in outlier:
    train_WO = remove_out(train, col)
print('Shape of Train Data after removing Outliers:', train_WO.shape)


# check variable type and data information
train.info()


# Remove unwanted columns
train = train.drop(['holiday', 'atemp', 'casual', 'registered'], axis = 1)
train_WO = train_WO.drop(['holiday', 'atemp', 'casual', 'registered'], axis = 1)


# Split data in Train & Test
from sklearn.model_selection import train_test_split

###########################
#                         #
#  train & test dataset   #
#     with outliers       #
#                         #
###########################

X = train.drop(['count'], axis = 1)
y = train['count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# log conversion
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# with outlier
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


###########################
#                         #
#  train & test dataset   #
#    without outliers     #
#                         #
###########################

X = train_WO.drop(['count'], axis = 1)
y = train_WO['count']
X_train_WO, X_test_WO, y_train_WO, y_test_WO = train_test_split(X, y, test_size = 0.2, random_state = 0)

# log conversion
y_train_WO_log = np.log1p(y_train_WO)
y_test_WO_log = np.log1p(y_test_WO)

# without outliers
print(X_train_WO.shape)
print(X_test_WO.shape)
print(y_train_WO.shape)
print(y_test_WO.shape)


##########################
#                        #
#   Eveluation Metrics   #
#         RMSLE          #
#                        #
##########################

# define function to compute rmsle
def rmsle(actual, pred):
    actual = np.exp(actual)
    pred = np.exp(pred)
    rmsle = np.sqrt(np.mean((np.log1p(actual) - np.log1p(pred))**2))
    return rmsle


################################
#                              #
#  Multiple Linear Regression  #
#        with outliers         #
#                              #
################################

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# create Multiple Linear Regression model
lr.fit(X_train, y_train_log)

# make prediction of Multiple Linear regression Model for Train & Test data
lr_train_pred = lr.predict(X_train)
lr_test_pred = lr.predict(X_test)

lr_tr_rmsle = round(rmsle(y_train_log, lr_train_pred),4)
lr_te_rmsle = round(rmsle(y_test_log, lr_test_pred),4)

print("Multiple Linear Regression RMSLE Score for Train & Test Data")
print(f'Linear Regression Train Dataset: RMSLE:', lr_tr_rmsle)
print(f'Linear Regression Test Dataset:  RMSLE:', lr_te_rmsle)



################################
#                              #
#  Multiple Linear Regression  #
#       without outliers       #
#                              #
################################


from sklearn.linear_model import LinearRegression
lr_WO = LinearRegression()


# create Multiple Linear Regression model
lr_WO.fit(X_train_WO, y_train_WO_log)

# make prediction of Multiple Linear regression Model for Train & Test data
lr1_train_pred = lr_WO.predict(X_train_WO)
lr1_test_pred = lr_WO.predict(X_test_WO)

lr1_tr_rmsle = round(rmsle(y_train_WO_log, lr1_train_pred),4)
lr1_te_rmsle = round(rmsle(y_test_WO_log, lr1_test_pred),4)

print("Multiple Linear Regression RMSLE Score for Train & Test Data")
print(f'Linear Regression Train Dataset: RMSLE:',lr1_tr_rmsle)
print(f'Linear Regression Test Dataset:  RMSLE:',lr1_te_rmsle)



#########################
#                       #
#  K-Nearest Neighbour  #
#                       #
#########################

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()

# create KNN model
knn.fit(X_train, y_train_log)

# make prediction of KNN Model for Train & Test data
knn_train_pred = knn.predict(X_train)
knn_test_pred = knn.predict(X_test)

knn_tr_rmsle = round(rmsle(y_train_log, knn_train_pred),4)
knn_te_rmsle = round(rmsle(y_test_log, knn_test_pred),4)

print("K Neareast Neighbour RMSLE Score for Train & Test Data")
print(f'K Neareast Neighbour Train Dataset: RMSLE:',knn_tr_rmsle)
print(f'K Neareast Neighbour Test Dataset:  RMSLE:',knn_te_rmsle)


#########################
#                       #
#     Decision Tree     #
#                       #
#########################

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()

# create Decision Tree Model
dt.fit(X_train, y_train_log)

# make prediction of Decision Tree Model for Train & Test Data
dt_train_pred = dt.predict(X_train)
dt_test_pred = dt.predict(X_test)

dt_tr_rmsle = round(rmsle(y_train_log, dt_train_pred),4)
dt_te_rmsle = round(rmsle(y_test_log, dt_test_pred),4)

print("Decision Tree RMSLE Score for Train & Test Data")
print(f'Decision Tree Train Dataset: RMSLE:',dt_tr_rmsle)
print(f'Decision Tree Test Dataset:  RMSLE:',dt_te_rmsle)



#########################
#                       #
#     Random Forest     #
#                       #
#########################

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

# create Random Forest Model
rf.fit(X_train, y_train_log)

# make prediction of Random Forest Model for Train & Test Data
rf_train_pred = rf.predict(X_train)
rf_test_pred = rf.predict(X_test)

rf_tr_rmsle = round(rmsle(y_train_log,rf_train_pred ),4)
rf_te_rmsle = round(rmsle(y_test_log, rf_test_pred),4)

print("Random Forest RMSLE Score for Train & Test Data")
print(f'Random Forest Train Dataset: RMSLE:',rf_tr_rmsle)
print(f'Random Forest Test Dataset:  RMSLE:',rf_te_rmsle)



##########################
#                        #
#   Tune Random Forest   #
#                        #
##########################

from sklearn.model_selection import GridSearchCV
rf2 = RandomForestRegressor(random_state=1)

params = [{'n_estimators' : [500, 600, 800],'max_features':['auto', 'sqrt'],
           'min_samples_split':[2,4,6],'max_depth':[12, 14, 16],'min_samples_leaf':[2,3,5],
           'random_state' :[1]}]

grid_search = GridSearchCV(estimator=rf2, param_grid=params,cv = 5, n_jobs=-1)

grid_search = grid_search.fit(X_train, y_train_log)
print(grid_search.best_params_)

rf22 = RandomForestRegressor(random_state=1,max_depth = 14, max_features = 'auto', min_samples_leaf = 2, 
                             min_samples_split = 2, n_estimators =600)

rf_final = rf22.fit(X_train, y_train_log)

rf_tr = rf_final.predict(X_train)
rf_te = rf_final.predict(X_test)

rf1_tr_rmsle = round(rmsle(y_train_log,rf_tr ),4)
rf1_te_rmsle = round(rmsle(y_test_log, rf_te),4)

print("Random Forest RMSLE Score for Train & Test Data")
print(f'Random Forest Train Dataset: RMSLE:',rf_tr_rmsle)
print(f'Random Forest Test Dataset:  RMSLE:',rf_te_rmsle)


################################
#                              #
#  Compare the performance of  #
#       all used models        #
#                              #
################################


# make data frame 'all_model' for store RMSE score of all used model
all_model = pd.DataFrame({"model": ['Multiple Linear Regression with outlier',
                                    'Multiple Linear Regression without outlier',
                                    'K-Neareast Neighbour',
                                    'Decision Tree',
                                    'Random Forest','Tune Random Forest'], 
                           "train_rmsle": [lr_tr_rmsle, lr1_tr_rmsle, knn_tr_rmsle, 
                                           dt_tr_rmsle, rf_tr_rmsle, rf1_tr_rmsle],
                         "test_rmsle": [lr_te_rmsle, lr1_te_rmsle, knn_te_rmsle,
                                         dt_te_rmsle, rf_te_rmsle, rf1_te_rmsle]},
                         columns = ['model','train_rmsle','test_rmsle'])
all_model = all_model.sort_values(by = 'test_rmsle', ascending = False)


# create a barplot for all_model
sns.barplot(all_model['test_rmsle'], all_model['model'], palette = 'Set2')
plt.xlabel('Root Mean Square Log Error')
plt.ylabel('Models')
plt.title('Compare performance of all used model')

print(all_model)

