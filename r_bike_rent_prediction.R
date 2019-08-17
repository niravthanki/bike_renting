# import librararies
library(ggplot2)
library(dplyr) 
library(randomForest) 
library(gridExtra) 
library(ggcorrplot)
library(usdm)
library(caret) 
library(moments)

#  Set Working Directory      
setwd("D:/Bike")

# Lodad Data Set         
train = read.csv('day.csv')

cat("Dimension of Train Data:", dim(train))

# head of train data
head(train)

# check variable type and data information
str(train)

summary(train)

# drop non useful columns 'instant' & 'dteday'
train = train %>%
  dplyr::select(-instant, -dteday)

# rename some columns for understanding
train = train %>%
  dplyr::rename(year = yr,
                month = mnth,
                weather = weathersit,
                humidity = hum,
                count = cnt)

# list of numerical columns & categorical columns
num_col = c('temp', 'atemp', 'humidity', 'windspeed', 'casual','registered', 'count')
cat_col = c('season', 'year', 'month', 'holiday', 'weekday', 'workingday','weather')

# statistical summary of numerical columns
summary(train[,num_col])

# unique values in categorical columns
rapply(train[,cat_col], function(x)length(unique(x)))


# count of unique categories in each columns
lapply(train[,cat_col], function(x)table(x))


###############################
#                             #
#      Missing Value          #   
#                             #
###############################


apply(train, 2, function(x){sum(is.na(x))})



# # Visualization of Numerical Variables
#################################
#                               #
#          Boxplot of           #
#     Numerical Variables       #
#                               #
#################################

box_plot = function(df, col){
  df$x = 1
  ggplot(aes_string(x = 'x', y = col), data = df)+
    geom_boxplot()+
    xlab(col)+
    ggtitle(paste("Box-Plot: ",col))
}

all_boxplot = lapply(c('temp','atemp','humidity','windspeed'), box_plot, df = train)

grid.arrange(all_boxplot[[1]],all_boxplot[[2]],all_boxplot[[3]],all_boxplot[[4]], ncol = 4)



#################################
#                               #
#        Histogram of           #
#     Numerical Variables       #
#                               #
#################################

histogram = function(df, col){
  ggplot(aes_string(col), data = df)+
    geom_histogram(fill = 'blue', color = 'black',bins = 10)+
    xlab(col)+
    ggtitle(paste("Histogram: ",col))

}

all_histogram = lapply(c('temp','atemp','humidity','windspeed'), histogram, df = train)

grid.arrange(all_histogram[[1]],all_histogram[[2]],all_histogram[[3]],all_histogram[[4]], 
             ncol = 4)

cat('Skewness of Temperature:',round(skewness(train$temp),2))
cat('Skewness of ATemperature:',round(skewness(train$atemp),2))
cat('Skewness of Humidity:',round(skewness(train$humidity),2))
cat('Skewness of Windspeed:',round(skewness(train$windspeed),2))


#################################
#                               #
#       Scatter plot of         #  
#     Numerical Variables       #
#                               #
#################################

scatter = function(df, col){
  ggplot(aes_string(y = 'count', x = col),data = df)+
    geom_point(color = 'blue')+
    geom_smooth(method = 'lm', se = FALSE)+
    xlab(col)+
    ggtitle(paste("Scatter Plot: ",col))
}


all_scatter = lapply(c('temp','atemp','humidity','windspeed'), scatter, df = train)

grid.arrange(all_scatter[[1]],all_scatter[[2]],all_scatter[[3]],all_scatter[[4]],ncol = 4)

cat('Correlation of Temperature with Target Variable:',round(cor(train$temp, train$count),2))
cat('Correlation of ATemperature with Target Variable:',round(cor(train$atemp, train$count),2))
cat('Correlation of Humidity with Target Variable:',round(cor(train$humidity, train$count),2))
cat('Correlation of Windspeed with Target Variable:',round(cor(train$windspeed, train$count),2))


# # Visualization of Categorical Variables
#################################
#                               #
#         Bar plot of           #  
#     Numerical Variables       #
#                               #
#################################

bar_plot = function(df, cat_col, val_col){
  
  cc <- enquo(cat_col)
  vc <- enquo(val_col)
  cc_name <- quo_name(cc) # generate a name from the enquoted statement!
  
  df <- df %>%
    dplyr::group_by(!!cc) %>%
    dplyr::summarise (mean_count = mean(!!vc)) %>%
    dplyr::mutate(!!cc_name := factor(!!cc, !!cc)) # insert pc_name here!
  
  ggplot(df) + aes_(y = ~mean_count, x = substitute(cat_col)) +
    geom_bar(stat="identity", width = 0.5)+
    ggtitle(paste("Bar Plot: ", substitute(cat_col)))
  
}

# create bar plot
year = bar_plot(train, year, count)
month = bar_plot(train, month, count)
weekday = bar_plot(train, weekday, count)


grid.arrange(year, month, weekday, ncol = 3)

season = bar_plot(train, season, count)
weather = bar_plot(train, weather, count)

grid.arrange(season, weather, ncol = 2)


# mean count of Month & weekday with respect of Year
month_yr = ggplot(train, aes(x = factor(month), y = count, fill = factor(year)))+
  stat_summary(fun.y = 'mean', geom = 'bar',position = 'dodge')

week_yr = ggplot(train, aes(x = factor(weekday), y = count, fill = factor(year)))+
  stat_summary(fun.y = 'mean', geom = 'bar',position = 'dodge')

grid.arrange(month_yr, week_yr, ncol = 2)


#################################
#                               #
#     Feature Engineering       # 
#                               #
#################################

# create function for month_bin
month_bin = function(df){
  df %>%
    mutate(month_bin = ifelse(month <= 4, 0,
                              ifelse(month >= 11, 0, 1))) %>%
    dplyr::select(-month)
}


# create function for weekday_bin
weekday_bin = function(df){
  df %>%
    mutate(weekday_bin = ifelse(weekday < 2, 0, 1)) %>%
    dplyr::select(-weekday)
}


# create two new variables 'month_bin' & 'weekday_bin' and drop original variabl
train = month_bin(train)
train = weekday_bin(train)


#################################
#                               #
#       Heat Map of             #
#   Numerical variables         #
#                               #
#################################

corr = round(cor(train[,6:12]),2)

ggcorrplot(corr, hc.order = TRUE, type = 'lower', lab = TRUE)


#################################
#                               #
#       Chi-Square Test         # 
#                               #
#################################

chisqmatrix <- function(x){
  names = colnames(x) 
  num = length(names)
  m = matrix(nrow=num,ncol=num,dimnames=list(names,names))
  for (i in 1:(num-1)) {
    for (j in (i+1):num) {
      m[i,j] = round(chisq.test(x[,i],x[,j],)$p.value,4)
    }
  }
  return (m)
}


chisquare_mat = chisqmatrix(train[,c(1,2,3,4,5,13,14)])
print(chisquare_mat)


#################################
#                               #
#      Feature Importance       # 
#                               #
#################################

# Check Feature importance with the help of Random Fores
rf = randomForest(count ~., data = train[,-c(10,11)], ntree = 300, importance = TRUE)


imp = data.frame(importance(rf, type = 1))
imp$features = rownames(imp)
names(imp)[1] = 'feature_imp'
rownames(imp) = NULL
imp = imp[,c(2,1)]
imp = imp[order(-imp$feature_imp),]
print(imp)


#################################
#                               #
#     Multi - Collinerity       # 
#                               #
#################################

# Check VIF score of Independent Numerical Variables (before removal)
vif(train[,c(6,7,8,9)])

# Check VIF score of Independent Numerical Variables (after removal)
vif(train[,c(6,8,9)])


#################################
#                               #
#    factor conversion of       #  
#      season & weather         #
#                               #
#################################

train[,c('season','weather')] = lapply(train[,c('season','weather')], as.factor)


# release memory
rm(list = setdiff(ls(),"train"))

# make copy of original dataframe
train_WO = train


#################################
#                               #
#    remove outliers form       #  
#    humidity & windspeed       #
#                               #
#################################

for (i in c('humidity', 'windspeed')){
  outlier = train_WO[,i] [train_WO[,i] %in% boxplot.stats(train_WO[,i])$out]
  train_WO = train_WO[which(!train_WO[,i] %in% outlier),]
}


# # Remove unwanted columns
colnames(train)

train = train %>%
  dplyr::select(-holiday, -atemp, -casual, - registered)

train_WO = train_WO %>%
  dplyr::select(-holiday, -atemp, -casual, - registered)


#################################
#                               #
#     Train & Test dataset      #  
#        (with outliers)        #
#                               #
#################################

set.seed(1)

sample = sample.int(n = nrow(train), size = floor(.80*nrow(train)), replace = FALSE)
train_1 = train[sample,]
test_1 = train[-sample,]


#################################
#                               #
#     Train & Test dataset      #  
#      (without outliers)       #
#                               #
#################################

sample_WO = sample.int(n = nrow(train_WO), size = floor(0.80*nrow(train_WO)), replace = FALSE)
train_WO_1 = train_WO[sample_WO,]
test_WO_1 = train_WO[-sample_WO,]


#################################
#                               #
#      Evaluation Metircs       #
#        RMSLE Function         #
#                               #
#################################

rmsle = function(actual, pred){
  actual = exp(actual)
  pred = exp(pred)
  sqrt(mean((log1p(actual) - log1p(pred))**2))
}


#################################
#                               #
#  Multiple Linear Regression   #    
#       (with outliers)         #
#                               #
#################################

# create Multiple Linear Regression model
lr_mod = caret::train(log1p(count) ~., data = train_1, method = 'lm')

# make prediction of Multiple Linear regression Model for Train & Test data
lr_train_pred = predict(lr_mod, train_1)
lr_test_pred = predict(lr_mod, test_1)

lr_tr_rmsle = round(rmsle(log1p(train_1$count), lr_train_pred),4)
lr_te_rmsle = round(rmsle(log1p(test_1$count), lr_test_pred),4)

cat("Linear Regression Train RMSLE", lr_tr_rmsle)
cat("Linear Regression Test RMSLE", lr_te_rmsle)


#################################
#                               #
#  Multiple Linear Regression   #    
#     (without outliers)        # 
#                               #
#################################

# create Multiple Linear Regression model
lr_mod_WO = caret::train(log1p(count) ~., data = train_WO_1, method = 'lm')

# make prediction of Multiple Linear regression Model for Train & Test data
lr1_train_pred = predict(lr_mod_WO, train_WO_1)
lr1_test_pred = predict(lr_mod_WO, test_WO_1)

lr1_tr_rmsle = round(rmsle(log1p(train_WO_1$count), lr1_train_pred),4)
lr1_te_rmsle = round(rmsle(log1p(test_WO_1$count), lr1_test_pred),4)

cat("Linear Regression Train RMSLE:", lr1_tr_rmsle)
cat("Linear Regression Test RMSLE:", lr1_te_rmsle)


#################################
#                               #
#    K Neareast Neighbour       #
#                               #
#################################

# create KNN model
knn_mod = caret::train(log1p(count) ~., data = train_1, method = 'knn')

# make prediction of KNN Model for Train & Test data
knn_train_pred = predict(knn_mod, train_1)
knn_test_pred = predict(knn_mod, test_1)

knn_tr_rmsle = round(rmsle(log1p(train_1$count), knn_train_pred),4)
knn_te_rmsle = round(rmsle(log1p(test_1$count), knn_test_pred),4)

cat("K-Nearest Neighbour Train RMSLE:", knn_tr_rmsle)
cat("K-Nearest Neighbour Test RMSLE:", knn_te_rmsle)


#################################
#                               #
#         Decision Tree         #
#                               #
#################################

# create Decision Tree Model
dt_mod = caret::train(log1p(count) ~., data = train_1, method = 'rpart2')

# make prediction of Decision Tree Model for Train & Test Data
dt_train_pred = predict(dt_mod, train_1)
dt_test_pred = predict(dt_mod, test_1)

dt_tr_rmsle = round(rmsle(log1p(train_1$count), dt_train_pred),4)
dt_te_rmsle = round(rmsle(log1p(test_1$count), dt_test_pred),4)

cat("Decision Tree Train RMSLE:", dt_tr_rmsle)
cat("Decision Tree Test RMSLE:", dt_te_rmsle)


#################################
#                               #
#    Random Forest Default      #
#                               #
#################################

# create Random Forest Model
rf_mod = caret::train(log1p(count) ~., data = train_1, method = 'rf')

# make prediction of Random Forest Model for Train & Test Data
rf_train_pred = predict(rf_mod, train_1)
rf_test_pred = predict(rf_mod, test_1)

rf_tr_rmsle = round(rmsle(log1p(train_1$count), rf_train_pred),4)
rf_te_rmsle = round(rmsle(log1p(test_1$count), rf_test_pred),4)

cat("Random Forest Train RMSLE:", rf_tr_rmsle)
cat("Random Forest Test RMSLE:", rf_te_rmsle)


#################################
#                               #
#       Tune Random Forest      #
#                               #
#################################

cnt = trainControl(method = 'repeatedcv', n = 5, repeats = 4)
rf1_mod = caret::train(log1p(count) ~., data = train_1, method = 'rf', trControl = cnt)

rf1_mod$bestTune

rf1_mod$finalModel

rf1_train_pred = predict(rf1_mod, train_1)
rf1_test_pred = predict(rf1_mod, test_1)

rf1_tr_rmsle = round(rmsle(log1p(train_1$count), rf1_train_pred),4)
rf1_te_rmsle = round(rmsle(log1p(test_1$count), rf1_test_pred),4)

cat('Tune Random Forest Train RMSLE', rf1_tr_rmsle)
cat('Tune Random Forest Test RMSLE', rf1_te_rmsle)


#################################
#                               #
#  Compare the performance of   #    
#       all used models         # 
#                               #
#################################

all_model = data.frame("model" = c("Multiple Linear Regression with outlier",
                                   "Multiple Linear Regression without outlier",
                                   "K Nearest Neighbour", "Decision Tree",
                                   "Random Forest", "Tune Random Forest"),
                       "train_rmsle" = c(lr_tr_rmsle, lr1_tr_rmsle, knn_tr_rmsle, 
                                         dt_tr_rmsle, rf_tr_rmsle,rf1_tr_rmsle),
                       "test_rmsle" = c(lr_te_rmsle, lr1_te_rmsle, knn_te_rmsle,
                                        dt_te_rmsle, rf_te_rmsle, rf1_te_rmsle))


all_model = all_model[order(all_model$test_rmsle),]

print(all_model)


# Bar graph of model performance
ggplot(data = all_model, aes(x = reorder(model, test_rmsle),test_rmsle))+
  geom_bar(stat = "identity", fill = "blue")+
  coord_flip()+
  ggtitle("Compare Performance of all used Models")+
  xlab("Models")+
  ylab("Root Mean Square Log Error")