

#List of libraries
install.packages("stringr")
install.packages("homals")
install.packages("VIM")
install.packages("naniar")
install.packages(c("FactoMineR", "factoextra"))
install.packages("devtools")
install.packages('MASS')
install.packages('Amelia')
install.packages('mice')
install.packages('ggparallel')
install.packages('gmodels')
install.packages('gridExtra')
install.packages('mlbench')
install.packages('caret')
install.packages('randomForest', dependences=TRUE)
install.packages('caTools')
install.packages("class")
install_github("vqv/ggbiplot", force=TRUE)
install.packages('ROCR')
install.packages('gridSVG')
install.packages('ipred')
remove.packages('ipred')

library(stringr)
library(homals)
library(VIM)
library(naniar)
library(FactoMineR)
library(factoextra)
library(devtools)
library(ggbiplot)
library(ggplot2)
library(MASS)
library(Amelia)
library(mice)
library(lattice)
library(ggparallel)
library(gmodels)
library(mlbench)
library(caret)
library(caTools)
library(gridExtra)
library(grid)
# Caret dependency can interfere with class library
library(class)
library(ROCR)


mushrooms<-read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"), header=F)
mushrooms_copy<-read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"), header=FALSE, stringsAsFactors = TRUE)


# Data prep



column_names<-c('class','cap-shape', 'cap-surface', 'cap-color', 'bruised', 'odor',
                'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 
                'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                'stalk-surface-below-ring', 'stalk-color-above-ring',
                'stalk-color below ring', 'veil type', 'veil color', 'ring number',
                'ring type', 'spore print color', 'population', 'habitat')

column_names<-str_replace_all(column_names, c(" "= "-"))
colnames(mushrooms)<-column_names
head(mushrooms)

summary(mushrooms)
dim(mushrooms)

str(mushrooms)

# Veil-type has only one categorical level used so can be removed
mushrooms$`veil-type`<-NULL


# Missing values stalk-root='?'
which(mushrooms$`stalk-root`=='?')
mushrooms[3985,12]


?replace_with_na_all
mushrooms[mushrooms=="?"]<- NA
gg_miss_var(mushrooms)

# Missng values
res<-summary(aggr(mushrooms, sortVar=TRUE))$combinations
?aggr
head(res[rev(order(res[,2])),])
pct_miss(mushrooms)
n_miss(mushrooms)
n_complete(mushrooms)
n_miss(mushrooms$`stalk-root`)
head(mushrooms)
mushroom_features<- mushrooms[-1]

# Imputation stalk-root through mice package - doesn't work
imp.mushrooms <- mice(mushrooms, m=1, method='cart' , printFlag=FALSE)
sort(sapply(imp.mushrooms, function(x) {sum(is.na(x))}), decreasing = TRUE)
anyNA(imp.mushrooms)

#Drop the column instead
mushrooms$`stalk-root`<- NULL

# Dummy values / encoded categories - Never used
dmy<- dummyVars(" ~ .", data = mushrooms, fullRank=T)
trsf <- data.frame(predict(dmy, newdata = mushrooms))
trsf
enc_features <- trsf[-1]
enc_outcomes<- trsf[1]

# Visualize features
ggparallel(list("class", relevant_features[1:5]), data=mushrooms)

# MCA, Component analysis
?MCA
res.mca<-MCA(mushroom_features)
res.mca
str(res.mca)
ggbiplot(res.mca, labels=rownames(mushrooms))

mca_vars_df <- data.frame(res.mca$var$coord)

eig.val<- get_eigenvalue(res.mca)
head(eig.val)
fviz_screeplot(res.mca, addlabels = TRUE, ylim = c(0,10))
fviz_mca_biplot(res.mca, ggtheme = theme_minimal())


#Chi-Square test for correlation between individual values and class
chisq_test_res = list()
relevant_features = c()
for(i in 2:length(colnames(mushrooms))) {
  fname = colnames(mushrooms)[i]
  res = chisq.test(mushrooms[,i], mushrooms[, "class"], simulate.p.value = TRUE)
  res$data.name = paste(fname, "class", sep= " and ")
  chisq_test_res[[fname]] = res
  relevant_features = c(relevant_features, fname)
}
chisq_test_res
relevant_features

# Seed used for all models
set.seed(10)

# KNN
mushrooms$class <- as.factor(mushrooms$class)
intrain <- createDataPartition(y = mushrooms$class, p=0.8, list=FALSE)
training <- mushrooms[intrain,]
# Counts instances after intrain
testing <- mushrooms[-intrain,]
actual_value<- testing$class

knn_fit <- train(class ~ ., method = "knn", data = training, trControl = trainControl(method = 'repeatedcv', number = 10, repeats = 3,classProbs = TRUE))
#Accuracy of 100% with k=5
print(knn_fit)
plot(knn_fit)
#Test Knn fit
test_predict <- predict(knn_fit, newdata = testing[,-1])
confusionMatrix(table(actual_value, test_predict))
typeof(test_predict)
pred_val<- prediction(df, testing$class)

plot(varImp((knn_fit), main="KNN - Impotance variable"))
dim(mushrooms)
?train
# Random forest
sample = sample.split(mushrooms$class, SplitRatio = .8)
x_train = subset(mushrooms, sample==T)
x_test = subset(mushrooms, sample == F)
nlevels(mushrooms$class)
y_train<- x_train$class
y_test<-x_test$class
x_train$class<-NULL
x_test$class<-NULL
# stratified sample
cv.10.folds<-createMultiFolds(y_train, k=10, time=2)


control <- trainControl(method="repeatedcv", number=10, repeats=2, index=cv.10.folds)


rf_fit<-train(x=x_train, y=y_train, method="rf", trControl = control, tuneLength = 3)
print(rf_fit)
plot(varImp((rf_fit), main="Random Forest - Variable Importance Plot"))

# Using test set
y_predicted<- predict(rf_fit, x_test)

rf_df<-data.frame(Expect=y_test,Pred=y_predicted)

confusionMatrix(table(rf_df$Expect, rf_df$Pred))

