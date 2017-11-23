library(dplyr)
library(stringr)
library(caret)
library(randomForest)
library(klaR)
library(tm)
library(lubridate)
library(Matrix)
library(xgboost)
library(e1071)
library(nnet)
setwd("~/Desktop/Rahul/ML3")
set.seed(100)
#===================Load Files ===========#
train1 <- read.csv('train.csv',header = TRUE,stringsAsFactors = FALSE)
test1 <- read.csv('test.csv',header = TRUE,stringsAsFactors = FALSE)
View(train1)

#sampledata = sample(seq_len(nrow(train1)),size=samplesize)
set.seed(100)
train1_bk <- train1
train1 <-train1_bk
train1 <- train1[1:(.5 * nrow(train1)),]
samplesize = (0.75 * nrow(train1))


#train1 <- train1[train1$browserid != '',]
#train1 <- train1[train1$devid != '',]
#======================Feature extraction ==============#

train1[train1$browserid=='Edge',]$devid <- 'Tablet'
train1[train1$browserid=='Safari',]$devid <- 'Tablet'
train1[train1$browserid=='Firefox',]$devid <- 'Mobile'
train1[train1$browserid=='Mozilla',]$devid <-  'Desktop'
train1[train1$browserid=='Mozilla Firefox',]$devid <-  'Desktop'
train1[train1$browserid=='InternetExplorer',]$devid <-  'Desktop'
train1[train1$browserid=='Internet Explorer',]$devid <-  'Tablet'
train1[train1$browserid=='Chrome',]$devid <-  'Desktop'
train1[train1$browserid=='Google Chrome',]$devid <-  'Mobile'
train1[train1$browserid=='Opera',]$devid <-  'Mobile'
train1[train1$browserid=='IE',]$devid <-  'Mobile'

train1[train1$browserid == '',]$browserid <- 'Unknown'
train1[train1$devid == '',]$devid <- 'Unknown'

browser.train1 <- data.frame(browserid=c('Mozilla Firefox','Edge','Chrome','InternetExplorer','Internet Explorer','Google Chrome','IE',
                                         'Opera','Safari','Mozilla','Firefox','Unknown'),newbrowserid=c('Firefox','MS','GOOGLE','MS','MS',
                                                                                                        'GOOGLE','MS','OPERA','Safari','Firefox','Firefox','Unknown'))

train1 <- train1 %>%
  left_join(browser.train1,by="browserid")



train1$date1 <- as.Date(train1$datetime)
train1$time1 <- strftime(train1$datetime,format="%H:%M:%S")
train1$dayofweek <- weekdays(as.Date(train1$date1))

day.weekend.train1 <- data.frame(dayofweek=c('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'),
                                 weekend=c('Weekday','Weekday','Weekday','Weekday','Weekday','Weekend','Weekend'))

train1 <- train1 %>%
  left_join(day.weekend.train1,by="dayofweek")

handheld.train1 <- data.frame(devid=c('Mobile','Tablet','Desktop','Unknown'),handheld=c('Yes','Yes','No','No'))
train1 <- train1 %>%
  left_join(handheld.train1,by="devid")

train1$ampm <- 'test'
train1$timehours <- hour(hms(train1$time1))
train1[train1$timehours >= 00 & train1$timehours <= 04,]$ampm <- 'Am'
train1[train1$timehours > 04 & train1$timehours <= 08,]$ampm <- 'Am1'
train1[train1$timehours > 08 & train1$timehours <= 12,]$ampm <- 'Am2'
train1[train1$timehours > 12 & train1$timehours <= 16,]$ampm <- 'pm1'
train1[train1$timehours > 16 & train1$timehours <= 20,]$ampm <- 'pm2'
train1[train1$timehours > 20 & train1$timehours <= 23,]$ampm <- 'pm'
#train1[train1$ampm != 'Am',]$ampm<- 'Pm'


train1$ampm <- as.factor(train1$ampm)
train1$countrycode <- as.factor(train1$countrycode)
train1$devid <- as.factor(train1$devid)

day.number.train1 <- data.frame(dayofweek=c('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'),
                                daynumber=c(1,2,3,4,5,6,7))

train1 <- train1 %>%
  left_join(day.number.train1,by="dayofweek")

train1[is.na(train1$siteid),]$siteid <- mean(train1[!is.na(train1$siteid),]$siteid)
a <- kmeans(data.frame(train1$offerid,train1$category,train1$daynumber,
                       (as.integer(as.factor(train1$countrycode))-1),(as.integer(as.factor(train1$weekend))-1)),40,iter.max = 20)
train1$newfeature <- a$cluster
train1$newfeature <- as.factor(train1$newfeature)

#=====================end of features ====================#

sampledata = sample(seq_len(nrow(train1)),size=samplesize)
train2 <- train1[sampledata,]
train3 <- train1[-sampledata,]

#===============predict using Logistic regression

myLogistic <- glm(as.numeric(click) ~  newbrowserid +   devid + countrycode  + weekend + handheld + ampm +  (merchant  
                                                                                                             * category * offerid) 
                  , data = train2
                  ,family = binomial(link = "logit"))

logisticprediction <- predict(myLogistic,newdata =train3,type = 'response')
predround1 <- round(logisticprediction,3)
table(predround1,train3$click)

data1 <- data.frame(predround1,train3$ID)

#===============predict using Naive Bayes

myNaives <- naiveBayes(as.numeric(click) ~  newbrowserid +   devid + countrycode  + weekend + handheld + ampm +  merchant  
                       + category + offerid
                       , data = train2)

bayesprediction <- predict(myNaives,newdata =train3,type = 'raw')
naive_prob <- bayesprediction[,2]
predround2 <- round(naive_prob)
table(predround2,train3$click)

data3 <- data.frame(train3$ID, predround2)

#===============predict using lightgbm =========================#
train1gbm <- train1
train1gbm$newbrowserid <- (as.integer(as.factor(train1gbm$newbrowserid))-1)
train1gbm$devid <- (as.integer(as.factor(train1gbm$devid))-1)
train1gbm$countrycode <- (as.integer(as.factor(train1gbm$countrycode))-1)
train1gbm$weekend <- (as.integer(as.factor(train1gbm$weekend))-1)
train1gbm$handheld <- (as.integer(as.factor(train1gbm$handheld))-1)
train1gbm$ampm <- (as.integer(as.factor(train1gbm$ampm))-1)
train1gbm$daynumber <- (as.integer(as.factor(train1gbm$daynumber))-1)
train1gbm$siteid <- (as.integer(as.factor(train1gbm$siteid))-1)
train1gbm$category <- (as.integer(as.factor(train1gbm$category))-1)
train1gbm$offerid <- (as.integer(as.factor(train1gbm$offerid))-1)
train1gbm$merchant <- (as.integer(as.factor(train1gbm$merchant))-1)
train1gbm$date1 <- (as.integer(as.factor(train1gbm$date1))-1)

train1gbm1 <- data.frame(train1gbm$newbrowserid,train1gbm$devid,train1gbm$countrycode,train1gbm$weekend,train1gbm$handheld,
                         train1gbm$ampm, train1gbm$daynumber,train1gbm$siteid,
                         train1gbm$category,train1gbm$offerid,train1gbm$merchant,train1gbm$date1,train1gbm$click)

train1gbm1 <- data.frame(train1$newbrowserid,train1$devid,train1$countrycode,train1$weekend,train1$handheld,
                         train1$ampm, train1$daynumber,train1$siteid,
                         train1$category,train1$offerid,train1$merchant,train1$click)



sampledata1 = sample(seq_len(nrow(train1gbm1)),size=samplesize)
train2gbm <- train1gbm1[sampledata1,]
train3gbm <- train1gbm1[-sampledata1,]

sparse_matrix_train2 <- sparse.model.matrix(train1gbm.click ~ .-1, data = train2gbm)
sparse_matrix_train3 <- sparse.model.matrix(train1gbm.click ~ .-1, data = train3gbm)

sparse_matrix_train2 <- sparse.model.matrix(train1.click ~ .-1, data = train2gbm)
sparse_matrix_train3 <- sparse.model.matrix(train1.click ~ .-1, data = train3gbm)

train2xgbmatrix <- xgb.DMatrix(data = sparse_matrix_train2,label =train2gbm$train1.click)
train3xgbmatrix <- xgb.DMatrix(data = sparse_matrix_train3,label =train3gbm$train1.click)

modelxgboost <- xgboost(data = sparse_matrix_train2, label = as.numeric(train2gbm$train1.click),max.depth = 50, 
                        eval_metric = "auc",eta = 0.1, nthread = 3,seed = 1, nround = 10, objective = "binary:logistic")

params1 <- list(booster = "gbtree", objective = "binary:logistic", eta=0.1, gamma=0, max_depth=50, 
                min_child_weight=1, subsample=1, colsample_bytree=1)

modelxgboost1 <- xgb.cv(params = params1, data = train2xgbmatrix, nround = 10, nfold = 5, showsd = T, stratified = T, 
                        print.every.n = 10, early.stop.round = 10, maximize = F )

xgboostprediction_new  <- xgb.train (params = params, data = train2xgbmatrix, nrounds = 5, 
                                     watchlist = list(val=train3xgbmatrix,train=train2xgbmatrix), 
                                     print.every.n = 10, early.stop.round = 10, maximize = F , eval_metric = "error")

xgboostprediction_new_prediction <- predict(xgboostprediction_new,train3xgbmatrix)
predround_new <- round(xgboostprediction_new_prediction)
table(predround_new,train3$click)

xgboostprediction <- predict(modelxgboost,newdata =sparse_matrix_train3,type = 'response')
predround <- round(xgboostprediction,3)
table(predround,train3$click)

data2 <- data.frame(predround,train3$ID)
#=======================end of xgb ========================#

#===============predict using neural networks ============#
train2gbm2 <- train2gbm
train2gbm2$train1.click <- NULL
mymodel_nnet_train <- nnet(sparse_matrix_train2,class.ind(train2gbm$train1.click)
                           , size=20, softmax=TRUE)
prediction_nnet <- predict(mymodel_nnet_train,newdata =sparse_matrix_train3,type = 'raw')
predround3 <- round(prediction_nnet)
table(predround3[,2],train3$click)
data3 <- data.frame(predround3[,2],train3$ID)

#=======================end of nnet ========================#

#==============model ensembling ==================#
#final_data <- data.frame(ID=as.character(''),prediction = 0)
final_data <- data1
final_data$ID <- data1$train3.ID
final_data$prediction <- ((data1$predround1 + data2$predround + data3$predround2 )/3)
final_data$prediction <- ((data1$predround1 + data2$predround )/2)
table(round(final_data$prediction),train3$click)
#==============end of model ensembling ==================#
params <- list(
  objective = 'binary',
  metric = 'auc',
  feature_fraction = 0.7,
  bagging_fraction = 0.5,
  max_depth = 10
)

model <- lgb.train(params = params
                   ,data = train3gbm
                   ,valids = list(valid = train3gbm)
                   ,learning_rate = 0.1
                   ,early_stopping_rounds = 40
                   ,eval_freq = 20
                   ,nrounds = 500
)


#====================test dataset ==============#
set.seed(100)
test1 <- read.csv('test.csv',header = TRUE,stringsAsFactors = FALSE)
test1[test1$browserid=='Edge',]$devid <- 'Tablet'
test1[test1$browserid=='Safari',]$devid <- 'Tablet'
test1[test1$browserid=='Firefox',]$devid <- 'Mobile'
test1[test1$browserid=='Mozilla',]$devid <-  'Desktop'
test1[test1$browserid=='Mozilla Firefox',]$devid <-  'Desktop'
test1[test1$browserid=='InternetExplorer',]$devid <-  'Desktop'
test1[test1$browserid=='Internet Explorer',]$devid <-  'Tablet'
test1[test1$browserid=='Chrome',]$devid <-  'Desktop'
test1[test1$browserid=='Google Chrome',]$devid <-  'Mobile'
test1[test1$browserid=='Opera',]$devid <-  'Mobile'
test1[test1$browserid=='IE',]$devid <-  'Mobile'

test1[test1$browserid == '',]$browserid <- 'Unknown'
test1[test1$devid == '',]$devid <- 'Unknown'
#============= Test features=========#

test1$date1 <- as.Date(test1$datetime)
test1$time1 <- strftime(test1$datetime,format="%H:%M:%S")
test1$dayofweek <- weekdays(as.Date(test1$date1))

day.weekend.test1 <- data.frame(dayofweek=c('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'),
                                weekend=c('Weekday','Weekday','Weekday','Weekday','Weekend','Weekend','Weekend'))

test1 <- test1 %>%
  left_join(day.weekend.test1,by="dayofweek")

handheld.test1 <- data.frame(devid=c('Mobile','Tablet','Desktop','Unknown'),handheld=c('Yes','Yes','No','No'))
test1 <- test1 %>%
  left_join(handheld.test1,by="devid")

test1$ampm <- 'test'
test1$timehours <- hour(hms(test1$time1))
test1[test1$timehours >= 00 & test1$timehours <= 06,]$ampm <- 'Am'
test1[test1$timehours > 06 & test1$timehours <= 12,]$ampm <- 'Am1'
test1[test1$timehours > 12 & test1$timehours <= 18,]$ampm <- 'pm1'
test1[test1$timehours > 18 & test1$timehours <= 23,]$ampm <- 'pm'

browser.test1 <- data.frame(browserid=c('Mozilla Firefox','Edge','Chrome','InternetExplorer','Internet Explorer','Google Chrome','IE',
                                        'Opera','Safari','Mozilla','Firefox','Unknown'),newbrowserid=c('Firefox','MS','GOOGLE','MS','MS',
                                                                                                       'GOOGLE','MS','OPERA','Safari','Firefox','Firefox','Unknown'))

test1 <- test1 %>%
  left_join(browser.test1,by="browserid")

day.number.test1 <- data.frame(dayofweek=c('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'),
                               daynumber=c(1,2,3,4,5,6,7))

test1 <- test1 %>%
  left_join(day.number.test1,by="dayofweek")


test1[is.na(test1$siteid),]$siteid <- median(test1[!is.na(test1$siteid),]$siteid)
b <- kmeans(data.frame(test1$offerid,test1$category,test1$merchant,test1$siteid),40,iter.max = 10)
test1$newfeature <- b$cluster
test1$newfeature <- as.factor(test1$newfeature)

testprediction <- predict(myLogistic,newdata =test1,type = 'response')
predround_lr <- testprediction
output <- data.frame(test1$ID,predround_lr)
write.table(output,"output5.txt",sep = ",",row.names =  FALSE,quote = FALSE)

#===============predict using lightgbm =========================#
test1gbm <- test1
test1gbm$newbrowserid <- (as.integer(as.factor(test1gbm$newbrowserid))-1)
test1gbm$devid <- (as.integer(as.factor(test1gbm$devid))-1)
test1gbm$countrycode <- (as.integer(as.factor(test1gbm$countrycode))-1)
test1gbm$weekend <- (as.integer(as.factor(test1gbm$weekend))-1)
test1gbm$handheld <- (as.integer(as.factor(test1gbm$handheld))-1)
test1gbm$ampm <- (as.integer(as.factor(test1gbm$ampm))-1)
test1gbm$newfeature <- (as.integer(as.factor(test1gbm$newfeature))-1)
test1gbm$daynumber <- (as.integer(as.factor(test1gbm$daynumber))-1)

test1gbm1 <- data.frame(test1gbm$newbrowserid,test1gbm$devid,test1gbm$countrycode,test1gbm$weekend,test1gbm$handheld,
                        test1gbm$ampm,test1gbm$daynumber,test1gbm$category,test1gbm$offerid,test1gbm$merchant)

sparse_matrix_test1 <- sparse.model.matrix( ~ .-1, data = test1gbm1)
xgboostprediction_test1 <- predict(modelxgboost,newdata =sparse_matrix_test1,type = 'response')
predround_test <- xgboostprediction_test1
data_test_xgb <- data.frame(test1$ID, predround_test)
write.table(data_test_xgb,"output6.txt",sep = ",",row.names =  FALSE,quote = FALSE)

#===============predict using Naive Bayes

bayesprediction_test <- predict(myNaives,newdata =test1,type = 'raw')
naive_prob_test <- bayesprediction_test[,2]
predround_test_naive <- naive_prob_test
data_test_naive <- data.frame(test1$ID, predround_test_naive)

#==============model ensembling test==================#
#final_data <- data.frame(ID=as.character(''),prediction = 0)
final_data_test <- data_test_xgb
final_data_test$ID <- data_test_xgb$test1.ID 
final_data_test$prediction <- ((data_test_xgb$predround_test + output$predround_lr + data_test_naive$predround_test_naive)/3)
final_data_test1 <- data.frame(final_data_test$ID,round(final_data_test$prediction,3))
write.table(final_data_test1,"output7.txt",sep = ",",row.names =  FALSE,quote = FALSE)
#==============end of model ensembling ==================#

