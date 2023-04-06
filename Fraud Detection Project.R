#############-------Fraud Detection Project-------#############

#install the required libraries
if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(reshape2)) install.packages("reshape2") 
if(!require(corrplot)) install.packages("corrplot") 
if(!require(kableExtra)) install.packages("kableExtra") 
if(!require(ROCR)) install.packages("ROCR") 
if(!require(PRROC)) install.packages("PRROC") 
if(!require(caret)) install.packages("caret") 
if(!require(rpart)) install.packages("rpart") 
if(!require(rpart.plot)) install.packages("rpart.plot") 
if(!require(randomForest)) install.packages("randomForest") 
if(!require(class)) install.packages("class") 
if(!require(caTools)) install.packages("caTools") 
if(!require(e1071)) install.packages("e1071") 
if(!require(gbm)) install.packages("gbm") 
if(!require(lightgbm)) install.packages("lightgbm") 
if(!require(xgboost)) install.packages("xgboost") 
if(!require(keras)) install.packages("keras") 
if(!require(tensorflow)) install.packages("tensorflow") 

#load the libraries
library(tidyverse)
library(reshape2) #to use melt()
library(corrplot)
library(kableExtra)
library(ROCR) #to use prediction() and performance
library(PRROC) #to use pr.curve()
library(caret)
library(rpart)
library(rpart.plot) #to use prp()
library(randomForest)
library(class) #to use knn()
library(caTools) #to use sample.split()
library(e1071) #to use svm()
library(gbm)
library(lightgbm)
library(xgboost)
library(keras)
library(tensorflow)

#clear environment
rm(list=ls())

#make reproducible
set.seed(1)

#set working directories
setwd("/users/zhaolong/downloads")

#read the dataset
cc<- read.csv("creditcard.csv")

#data exploratory
dim(cc)
str(cc)
summary(cc)
head(cc,2)
table(cc$Class)

#find missing values
sapply(cc, function(x) sum(is.na(x)))


#Proportions between Legal and Frauds Transactions
cc %>%
  ggplot(aes(Class)) +
  geom_bar() +  
  scale_x_discrete() +
  labs(title = "Proportions between Legal and Frauds Transactions")

#Frequency of Transaction Time
cc %>%
  ggplot(aes(Time)) + 
  theme_minimal()  +
  geom_histogram(binwidth = 10)  +
  labs(title = "Frequency of Transaction Time")

#Fraud Frequency by Time
cc[cc$Class==1,] %>% group_by(Time) %>%
  summarize(count=n()) %>%
  ggplot(aes(Time, count)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Fraud Frequency by Time")

#Legal Frequency by Time
cc[cc$Class==0,] %>% group_by(Time) %>%
  summarize(count=n()) %>%
  ggplot(aes(Time, count)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Legal Frequency by Time")


#Frauds Amounts Distributions
cc%>%
  ggplot(aes(Amount)) + 
  geom_histogram(binwidth = 100) +
  labs(title = "Transaction Amounts Distributions")
  

#Top 10 Transaction Amounts
cc%>%
  group_by(Amount) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  head(n=10) %>% kable() %>% kable_styling(latex_options = "HOLD_position",bootstrap_options = c("responsive"),
                                           position = "center",
                                           full_width = FALSE) %>% column_spec(2,color = "white" , background ="green")

#Bottom 10 Transaction Amounts
cc%>%
  group_by(Amount) %>%
  summarise(count = n()) %>%
  arrange(count) %>%
  head(n=10) %>% kable() %>% kable_styling(latex_options = "HOLD_position",bootstrap_options = c("responsive"),
                                       position = "center",
                                       full_width = FALSE) %>% column_spec(1,color = "white" , background ="red")


#scale Amount column
cc$Amount <- scale(cc$Amount, center = TRUE, scale = TRUE)

#prepare Class for correlation
cc$Class <- as.numeric(cc$Class)

#plot the correlation matrix
cormat<-round(cor(cc),2)
corrplot(cormat, method = "color")


# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
upper_tri <- get_upper_tri(cormat)

# Melt the correlation matrix
library(reshape2)
melted_cormat <- melt(upper_tri, na.rm = TRUE)

# Plot the heatmap
ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

#data preparation and splitting
cc1<-cc[,-1]
index<-sample(1:nrow(cc1),as.integer(0.75*nrow(cc1)))
train <- cc1[index,]
test <- cc1[-index,]


##---------------------------logistic regression---------------------------
m1<-glm(Class~., train,family="binomial")

#apply the model on test dataset to predict
p0<-predict(m1, test, type = "response")

p1 <- ifelse(p0>0.5, 1, 0)


pred1<-prediction(p1,test$Class)


auc1 <- performance(pred1, "auc")
aucpr1 <- pr.curve(
  scores.class0 = p1[test$Class == 1], 
  scores.class1 = p1[test$Class == 0],
  curve = T,  
)
# have auc and aucpr plots
plot(performance(pred1, 'sens', 'spec'), main=paste("AUC:", auc1@y.values))
plot(aucpr1)

# add values to the result dataframe
results <- data.frame(Model = "Logistic Regression", AUC = auc1@y.values[[1]],
                      AUCPR = aucpr1$auc.integral)
# show results 
results %>%
  kable() %>%
  kable_styling(latex_options = "HOLD_position",bootstrap_options = "responsive",
                position = "center",
                full_width = FALSE) %>% 
  column_spec(2,color = "white" , background ="blue") %>% 
  column_spec(3,color = "white" , background ="red")
  


##---------------------------decision tree---------------------------
m2<-rpart(Class ~ ., data = train, method = "class", minbucket = 10)

#check the most important variables
prp(m2)

#apply the model on test dataset to predict
p2<-predict(m2, test, type = "class")

p2<-as.numeric(p2)
pred2<-prediction(p2,test$Class)

auc2 <- performance(pred2, "auc")
aucpr2 <- pr.curve(
  scores.class0 = p2[test$Class == 1], 
  scores.class1 = p2[test$Class == 0],
  curve = T,  
)
# have auc and aucpr plots
plot(performance(pred2, 'sens', 'spec'), main=paste("AUC:", auc2@y.values))
plot(aucpr2)

# add values to the result dataframe
results <- results %>% add_row(
  Model = "Decision Tree", 
  AUC = auc2@y.values[[1]],
  AUCPR = aucpr2$auc.integral
)

# show results 
results %>%
  kable() %>%
  kable_styling(latex_options = "HOLD_position",bootstrap_options = "responsive",
                position = "center",
                full_width = FALSE) %>% 
  column_spec(2,color = "white" , background ="blue") %>% 
  column_spec(3,color = "white" , background ="red")



##---------------------------random forest---------------------------

m3<-randomForest(Class ~ ., data = train, ntree = 30)

#check the most significant varaibles
data.frame(importance(m3)) %>% arrange(desc(IncNodePurity)) %>%
  kable() %>%
  kable_styling(bootstrap_options = "responsive",
                position = "center",
                full_width = FALSE)

#apply the model on test dataset to predict
p3<-predict(m3,test)

p3<-as.numeric(p3)
pred3<-prediction(p3,test$Class)
auc3 <- performance(pred3, "auc")
aucpr3 <- pr.curve(
  scores.class0 = p3[test$Class == 1], 
  scores.class1 = p3[test$Class == 0],
  curve = T,  
)

# have auc and aucpr plots
plot(performance(pred3, 'sens', 'spec'), main=paste("AUC:", auc3@y.values))
plot(aucpr3)

# add values to the result dataframe
results <- results %>% add_row(
  Model = "Random Forest", 
  AUC = auc3@y.values[[1]],
  AUCPR = aucpr3$auc.integral
)

# show results 
results %>%
  kable() %>%
  kable_styling(latex_options = "HOLD_position",bootstrap_options = "responsive",
                position = "center",
                full_width = FALSE) %>% 
  column_spec(2,color = "white" , background ="blue") %>% 
  column_spec(3,color = "white" , background ="red")



##---------------------------knn model---------------------------

m4 <- knn(train[,-30], test[,-30], train$Class, k=4, prob = TRUE)

p4<-as.numeric(m4)

# compute the AUC and AUCPR values
pred4<-prediction(p4,test$Class)

auc4 <- performance(pred4, "auc")
aucpr4 <- pr.curve(
  scores.class0 = p4[test$Class == 1], 
  scores.class1 = p4[test$Class == 0],
  curve = T,  
)

# have auc and aucpr plots
plot(performance(pred4, 'sens', 'spec'), main=paste("AUC:", auc4@y.values))
plot(aucpr4)

# add values to the result dataframe
results <- results %>% add_row(
  Model = "KNN Model", 
  AUC = auc4@y.values[[1]],
  AUCPR = aucpr4$auc.integral
)

# show results 
results %>%
  kable() %>%
  kable_styling(latex_options = "HOLD_position",bootstrap_options = "responsive",
                position = "center",
                full_width = FALSE) %>% 
  column_spec(2,color = "white" , background ="blue") %>% 
  column_spec(3,color = "white" , background ="red")



##---------------------------svm model---------------------------

#split 10% of cc1 into cc5 dataset
split5<-sample.split(cc1$Class, SplitRatio=0.1)
cc5<-subset(cc1, split5 == T)

#have the train and test data for svm model
index5<-sample(1:nrow(cc5),as.integer(0.75*nrow(cc5)))
train5 <- cc5[index5,]
test5 <- cc5[-index5,]


m5<-svm(Class ~ ., data = train5, kernel='sigmoid')

#apply the model on test dataset to predict
p5<-predict(m5,test5)

p5<-as.numeric(p5)
pred5<-prediction(p5,test5$Class)
# compute the AUC and AUCPR values
auc5 <- performance(pred5, "auc")
aucpr5 <- pr.curve(
  scores.class0 = p5[test5$Class == 1], 
  scores.class1 = p5[test5$Class == 0],
  curve = T,  
)

# have auc and aucpr plots
plot(performance(pred5, 'sens', 'spec'), main=paste("AUC:", auc5@y.values))
plot(aucpr5)

# add values to the result dataframe
results <- results %>% add_row(
  Model = "SVM Model", 
  AUC = auc5@y.values[[1]],
  AUCPR = aucpr5$auc.integral
)

# show results 
results %>%
  kable() %>%
  kable_styling(latex_options = "HOLD_position",bootstrap_options = "responsive",
                position = "center",
                full_width = FALSE) %>% 
  column_spec(2,color = "white" , background ="blue") %>% 
  column_spec(3,color = "white" , background ="red")



##---------------------------gbm model---------------------------

m6<- gbm(as.character(Class) ~ .,
                 distribution = "bernoulli", 
                 data = rbind(train, test), 
                 n.trees = 500,
                 interaction.depth = 3, 
                 n.minobsinnode = 100, 
                 shrinkage = 0.01, 
                 train.fraction = 0.75,
)

# Determine best iteration based on test data
best.iter = gbm.perf(m6, method = "test")

# Get feature importance
summary(m6, n.trees = best.iter)

# Make predictions based on this model
p6 = predict.gbm(
  m6, 
  newdata = test, 
  n.trees = best.iter, 
  type="response"
)

p6<-as.numeric(p6)
pred6<-prediction(p6,test$Class)

# compute the AUC and AUCPR values

auc6 <- performance(pred6, "auc")
aucpr6 <- pr.curve(
  scores.class0 = p6[test$Class == 1], 
  scores.class1 = p6[test$Class == 0],
  curve = T,  
)

# have auc and aucpr plots
plot(performance(pred6, 'sens', 'spec'), main=paste("AUC:", auc6@y.values))
plot(aucpr6)

# add values to the result dataframe
results <- results %>% add_row(
  Model = "GBM Model", 
  AUC = auc6@y.values[[1]],
  AUCPR = aucpr6$auc.integral
)

# show results 
results %>%
  kable() %>%
  kable_styling(latex_options = "HOLD_position",bootstrap_options = "responsive",
                position = "center",
                full_width = FALSE) %>% 
  column_spec(2,color = "white" , background ="blue") %>% 
  column_spec(3,color = "white" , background ="red")




##---------------------------lightgbm---------------------------

# make the training and testing data, and train the model
m7train <- lgb.Dataset(
  as.matrix(train[, colnames(train) != "Class"]), 
  label = as.numeric(as.character(train$Class))
)

m7test <- lgb.Dataset(
  as.matrix(test[, colnames(test) != "Class"]), 
  label = test$Class
)

m7p = list(
  objective = "binary", 
  metric = "binary_error"
)

m7 <- lgb.train(
  params = m7p, 
  data = m7train, 
  valids = list(test = m7test), 
  learning_rate = 0.01, 
  nrounds = 500,
  early_stopping_rounds = 40, 
  eval_freq = 20
)

# Get feature importance
lgb.importance(m7, percentage = TRUE) %>%
  kable() %>%
  kable_styling(bootstrap_options = "responsive",
                position = "center",
                full_width = FALSE)

# Make predictions based on this model
p7 = predict(
  m7, 
  data = as.matrix(test[, colnames(test) != "Class"]), 
  n = m7$best_iter)

# compute the AUC and AUCPR values
pred7 <- prediction(
  p7, 
  test$Class
)


auc7 <- performance(pred7, "auc")
aucpr7 <- pr.curve(
  scores.class0 = p7[test$Class == 1], 
  scores.class1 = p7[test$Class == 0],
  curve = T,
)

# have auc and aucpr plots
plot(performance(pred7, 'sens', 'spec'), main=paste("AUC:", auc7@y.values))
plot(aucpr7)

# add values to the result dataframe
results <- results %>% add_row(
  Model = "LightGBM Model", 
  AUC = auc7@y.values[[1]],
  AUCPR = aucpr7$auc.integral
)

# show results 
results %>%
  kable() %>%
  kable_styling(latex_options = "HOLD_position",bootstrap_options = "responsive",
                position = "center",
                full_width = FALSE) %>% 
  column_spec(2,color = "white" , background ="blue") %>% 
  column_spec(3,color = "white" , background ="red")




##---------------------------xgboost---------------------------

# make the training and testing data, and train the model
m8train <- xgb.DMatrix(as.matrix(train[, colnames(train) != "Class"]), 
                              label = as.numeric(as.character(train$Class)))

m8test <- xgb.DMatrix(as.matrix(test[, colnames(test) != "Class"]), 
                             label = as.numeric(as.character(test$Class)))

m8 <- xgb.train(
  data = m8train, 
  params = list(objective = "binary:logistic", 
                eta = 0.1, 
                max.depth = 3, 
                nthread = 6, 
                eval_metric = "aucpr"), 
  watchlist = list(test = m8test), 
  nrounds = 500, 
  early_stopping_rounds = 40,
  print_every_n = 20
)

# check the importance of the predictors
xgb.plot.importance(xgb.importance(colnames(train$Class),m8))


p8<-predict(m8, 
  newdata = as.matrix(test[, colnames(test) != "Class"]), 
  ntreelimit = m8$best_ntreelimit
)


# compute the AUC and AUCPR values
pred8<-prediction(p8,test$Class)

auc8 <- performance(pred8, "auc")
aucpr8 <- pr.curve(
  scores.class0 = p8[test$Class == 1], 
  scores.class1 = p8[test$Class == 0],
  curve = T,  
)

# have auc and aucpr plots
plot(performance(pred8, 'sens', 'spec'), main=paste("AUC:", auc8@y.values))
plot(aucpr8)

# add values to the result dataframe
results <- results %>% add_row(
  Model = "Xgboost Model", 
  AUC = auc8@y.values[[1]],
  AUCPR = aucpr8$auc.integral
)

# show results 
results %>%
  kable() %>%
  kable_styling(latex_options = "HOLD_position",bootstrap_options = "responsive",
                position = "center",
                full_width = FALSE) %>% 
  column_spec(2,color = "white" , background ="blue") %>% 
  column_spec(3,color = "white" , background ="red")


##---------------------------neural network---------------------------

X_train <- train[,-30]
X_test <- test[,-30]

y_train <- train[,30]
y_test <- test[,30]

m9 <- keras_model_sequential() %>%
  layer_dense(units = 64, kernel_initializer = "uniform", activation = "relu",
              input_shape = ncol(X_train)) %>%
  layer_dense(units = 32, kernel_initializer = "uniform", activation = "relu") %>%
  layer_dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid")

compile9<-m9 %>% compile(optimizer = "adam",
               loss = "binary_crossentropy", 
               metrics = c("binary_accuracy"))

compile9

kerasfit9 <- fit(object = m9,
               x = as.matrix(X_train),
               y = as.numeric(y_train),
               batch_size = 100,
               epochs = 10,
               validation_split = 0.25)
kerasfit9

plot(kerasfit9) + labs(title = "Deep Learning Training Result")


pp9<-predict(object = m9,
              x = as.matrix(X_test),batch_size = 200,verbose = 0) %>% as.vector()
head(pp9,3)
range(pp9)

p9 <- ifelse(pp9>0.5, 1, 0)
table(p9)

#p9a<-m9 %>% predict(as.matrix(X_test)) %>% `>`(0.5) %>% k_cast("int32")  %>% as.vector() 
#table(p9a)
#table(y_test)

# compute the AUC and AUCPR values
pred9<-prediction(p9,as.numeric(y_test))

auc9 <- performance(pred9, "auc")
aucpr9 <- pr.curve(
  scores.class0 = p9[test$Class == 1], 
  scores.class1 = p9[test$Class == 0],
  curve = T,  
)

# have auc and aucpr plots
plot(performance(pred9, 'sens', 'spec'), main=paste("AUC:", auc9@y.values))
plot(aucpr9)

# add values to the result dataframe
results <- results %>% add_row(
  Model = "Neutral Network", 
  AUC = auc9@y.values[[1]],
  AUCPR = aucpr9$auc.integral
)

# show results 
results %>%
  kable() %>%
  kable_styling(latex_options = "HOLD_position",bootstrap_options = "responsive",
                position = "center",
                full_width = FALSE) %>% 
  column_spec(2,color = "white" , background ="blue") %>% 
  column_spec(3,color = "white" , background ="red")


