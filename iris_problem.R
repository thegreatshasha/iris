rm(list=ls())
setwd('/Users/dola/Documents/WORK/data_science/learnds/IRIS_dataset/')
iris_data<-read.csv('iris.data',header=FALSE)
iris_name<-read.csv('iris.names',header=FALSE)
head(iris_data)

hist(iris_data$V1)
hist(iris_data$V2)
hist(iris_data$V3)
hist(iris_data$V4)

iris_cont<-iris_data[,1:4]
iris_sp<-iris_data[,5]

iris_pca<-princomp(iris_cont,center=TRUE,scale=TRUE)
summary(iris_pca)
biplot(iris_pca,choices=1:2,scale=1)

library(ggfortify)
autoplot(iris_pca)

plot1<-ggplot(data=iris_data)+geom_bar(aes(x=V4,fill=V5),binwidth=V4)
plot1

#1st way to make decision trees 
library(partykit)
iris_training<-iris_data[1:100,]
iris_tree<-ctree(V5~V1+V2+V3+V4,data=iris_data)
plot(iris_tree)
summary(iris_tree)

#2nd way to make decision trees
library(rpart)
iris_tree2<-rpart(V5~V1+V2+V3+V4,data=iris_data)
plot(iris_tree2)
text(iris_tree2)
summary(iris_tree2)
