output:
#HarvardX: PH125.9x Data Science 
#MovieLens Rating Preduction Project
#April 10, 2019
#Philip Ng

##I. Introduction

##II. Executive Summary

##III. Prepare Data

##edx dataset
##The following edx sex is used to perform the analysis in this project. The "ggplot2" package is also added to the edx set. 
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId], title = as.character(title), genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
validation <- temp %>%
semi_join(edx, by = "movieId") %>%
semi_join(edx, by = "userId")
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

##Training and Testing dataset
##The test and training datasets are derived using edx set: 80% sample for training, and 20% sample for testing.
set.seed(1)
train_index <- createDataPartition(y = edx$rating, times = 1, p = 0.8, list = FALSE)
train_set <- edx[train_index,]
temp <- edx[-train_index,]
test_set <- temp %>%
semi_join(train_set, by = "movieId") %>%
semi_join(train_set, by = "userId")
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)
rm(temp, removed)

##IV. Summarize Dataset
##The following ways are used to look at the raw data from different perspectives: shape, size, type, general layout. Inspecting data helps build up intuition and identify questions for the edx and validation datasets.

summary(edx)
str(edx)
dim(edx)
summary(validation)
str(validation)
dim(validation)

##Below are the top 10 geneneres

edx %>% separate_rows(genres, sep = "\\I")%>%
group_by(genres) %>%
summarize(count = n()) %>%
arrange(desc(count))

##Below are the top 10 movies

edx %>% group_by(movieId, title)%>%
summarize(count = n()) %>%
arrange(desc(count))

##Below are the top 10 movies by rating

edx %>% group_by(rating, title)%>%
summarize(count = n()) %>%
arrange(desc(count))


##V. Visualize Dataset

Data visualization is perhaps the fastest and most useful way to summarize and learn more about the data.

Visualization means creating charts and plots from the raw data. Plots of the distribution or spread of attributes can help spot outliers, strange or invalid data.

Rating Distribution: Users give full-star ratings more frequently than half-star ratings.

edx %>%
ggplot(aes(rating)) +
geom_histogram(binwidth = 0.5, color = "black") +
scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
ggtitle("Rating distribution")


##Rating No. Per User: A lot of users actually rate hundreds of movies. 

edx %>% count(userId) %>%
ggplot(aes(n)) +
geom_histogram(bins = 30, color = "black") +
scale_x_log10() +
xlab("Number of ratings") +
ylab("Number of users") +
ggtitle("Number of ratings per user")


Rating No. Per Movie: Most movies were rated hundreds or even thousands of times. 

edx %>%
count(movieId) %>%
ggplot(aes(n)) +
geom_histogram(bins = 30, color = "black") +
xlab("Number of ratings") +
ylab("Movie count") +
scale_x_log10() +
ggtitle("Number of ratings per movie")


##Mean Movie Rating Per User: After shortlisting those users that have rated at least 100 movies, it is found that most users gave ratings of 3.0, 3.5 and 4.0. 

edx %>%
group_by(userId) %>%
filter(n() >= 100) %>%
summarise(mean_rating = mean(rating)) %>%
ggplot(aes(mean_rating)) +
geom_histogram(bins = 30, color = "black") +
xlab("Mean movie rating") +
ylab("Number of users") +
ggtitle("Mean movie rating per user") +
scale_x_discrete(limits = c(seq(0.5,5,0.5)))

##VI. Evaluate Algoritum

##The following RMSE function is used to assess three alogritums in this section.

RMSE <- function(true_ratings, predicted_ratings){
sqrt(mean((true_ratings - predicted_ratings)^2))
}


##The 1st model predicts rating using Simple Average.

mu_hat <- mean(train_set$rating)
model_1_rmse <- RMSE(test_set$rating, mu_hat)
rmse_results <- data_frame(Model = "Simple Average", RMSE = model_1_rmse)
rmse_results%>%knitr::kable()


##The 2nd model considers movie effect by using the least square estimate.

mu <- mean(train_set$rating)
movie_avgs <- train_set %>%
group_by(movieId) %>%
summarize(b_i = mean(rating - mu))
predicted_ratings <- mu + test_set %>%
left_join(movie_avgs, by='movieId') %>%
.$b_i
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
data_frame(Model="Movie_Effect",
RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()


##The 3rd model considers user effect. 

##It is found that there is substantial variability across users: some users rate many movies while others are selective.

train_set %>%
group_by(userId) %>%
summarize(b_u = mean(rating)) %>%
filter(n()>=100) %>%
ggplot(aes(b_u)) +
geom_histogram(bins = 30, color = "black")

##User-specific effect model : lm(rating ~ as.factor(movieId) + as.factor(userId))
##We will compute an approximation instead for the reasons described earlier in 2nd model
user_avgs <- test_set %>%
left_join(movie_avgs, by='movieId') %>%
group_by(userId) %>%
summarize(b_u = mean(rating - mu - b_i))

##We can now construct predictors and see how much the RMSE improves
predicted_ratings <- test_set %>%
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
mutate(pred = mu + b_i + b_u) %>%
.$pred
model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
data_frame(Model="Movie + User_Effect",
RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()

##VII. Evaluate validation set

##Based on the results from the preceding section, the best resulting model, "Movie + User_Effect Model", is ran directly on the validation set to predict the movie ratings. It is found that the RMSE of the validation set is 0.8294.

user_avgs_validation <- validation %>%
left_join(movie_avgs, by='movieId') %>%
group_by(userId) %>%
summarize(b_u = mean(rating - mu - b_i))
predicted_ratings <- validation %>%
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs_validation, by='userId') %>%
mutate(pred = mu + b_i + b_u) %>%
.$pred
model_rmse_validation <- RMSE(predicted_ratings, validation$rating)
model_rmse_validation
```

##VIII. Conclusion

##THANKS FOR REVIEWING!
