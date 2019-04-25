##HarvardX: PH125.9x Data Science
##Capstone Project: Predict Diabetes Positiveness Using Linear Discriminant Analysis Model 
##April26, 2019
##Philip K W Ng

##I. Executive Summary


##II. Download R Package 

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(klaR)) install.packages("klaR", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(MASS)) install.packages("MASS", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(ggplot2)
library(caTools)
library(klaR)
library(data.table)
library(dplyr)
library(broom)
library(MASS)
library(corrplot)
theme_set(theme_classic())


##III. Download and Summarize Dataset 

data("PimaIndiansDiabetes2", package = "mlbench")
PimaIndiansDiabetes2 <- na.omit(PimaIndiansDiabetes2)
model <- glm(diabetes ~., data = PimaIndiansDiabetes2,
             family = binomial)
probabilities <- predict(model, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")


mydata <- PimaIndiansDiabetes2 %>%
  dplyr::select_if(is.numeric)
predictors <- colnames(mydata)

dim(mydata)

summary(mydata)

str(mydata)


##IV. Analyze Dataset

##Barplots

par(mfrow=c(2,4))
for(i in 1:8) {
  counts <- table(mydata[,i])
  name <- names(mydata)[i]
  barplot(counts, main=name)
}

##Correlation
correlations <- cor(mydata[,1:8])
corrplot(correlations, method="number")

##Scatter plot
pairs(mydata)

##ggplot
mydata <- mydata %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)
ggplot(mydata, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") +
  theme_bw() +
  facet_wrap(~predictors, scales = "free_y")

##Cook's distance 
plot(model, which = 4, id.n = 3)

##Top 3 largest values

model.data <- augment(model) %>%
  mutate(index = 1:n())
model.data %>% top_n(3, .cooksd)

##ggplot
ggplot(model.data, aes(index, .std.resid)) +
  geom_point(aes(color = diabetes), alpha = .5) +
  theme_bw()

##Filter

model.data %>%
  filter(abs(.std.resid) > 3)
car::vif(model)


##V. Build Classification Model

pima.data <- na.omit(PimaIndiansDiabetes2)

##Split the data into training and test set
set.seed(123)
training.samples <- pima.data$diabetes %>%
  createDataPartition(p=0.75, list=FALSE)
train.data <- pima.data[training.samples, ]
test.data <- pima.data[-training.samples, ]


##VI. Make Prediction

fit <- lda(diabetes ~., data = train.data)
# Make predictions on the test data
predictions <- predict(fit, test.data)
prediction.probabilities <- predictions$posterior[,2]
predicted.classes <- predictions$class
observed.classes <- test.data$diabetes

accuracy <- mean(observed.classes == predicted.classes)
accuracy

error <- mean(observed.classes != predicted.classes)
error


##VI. Measure Prediction Accuracy

# Confusion matrix

table(observed.classes, predicted.classes) %>%
  prop.table() %>% round(digits = 3)

confusionMatrix(predicted.classes, observed.classes,
                positive = "pos")


##VIII. Conclusion



##THANK YOU FOR REVIEWING!!

