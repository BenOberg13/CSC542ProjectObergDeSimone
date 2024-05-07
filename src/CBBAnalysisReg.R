rm(list = ls())
#regular season model
cbbr <- rbind(cbb13,cbb14,cbb15,cbb16,cbb17,cbb18,cbb19,cbb21,cbb22,cbb23)
is.na(cbbr)
summary(cbbr)
#remove names
cbbr <- cbbr[-1, ]
cbbr[] <- sapply(cbbr, as.numeric)
cbbr <- cbbr[,-c(1,2,3,22,23,21)]
num_null_values <- colSums(is.na(cbbr))
print(num_null_values)
cbbr <- na.omit(cbbr)
summary(cbbr)
library(dplyr)
library(caret)
set.seed(123)
#data vis
#corr
library(corrplot)
cbb_eda <- cbbr
cbb_mat <- cor(cbb_eda)
corrplot(cbb_mat,method = "circle")
#wins
plot(cbb13$V4)
#adj off eff
plot(cbb13$V5)
#barthag
plot(cbb13$V7)
#EFG_O
plot(cbb13$V8)
#two pt %
plot(cbb13$V16)
#corr to wins or V4
correlations <- cor(cbbr[, "V4"], cbbr)
print(correlations)
#create splits
train_percentage <- 0.8
train_indices <- createDataPartition(cbbr$V4, p = train_percentage, list = FALSE)


#remove team name and conference
#regular season model
class(cbbr$V4)
summary(cbbr)
train_set_r <- cbbr[train_indices, ]
test_set_r <- cbbr[-train_indices, ]
class(train_set_r$V4)
#print out number of training and testing examples
num_train_examples_r <- nrow(train_set_r)
num_test_examples_r <- nrow(test_set_r)
cat("Number of training examples:", num_train_examples_r, "\n")
cat("Number of testing examples:", num_test_examples_r, "\n")
colnames(test_set_r)
summary(test_set_r)

#random forest
library(randomForest)
control <- trainControl(method = "cv", number = 10)
# Build a random forest model with k-fold cross-validation and feature selection
rf_model_cv_fs <- train(V4 ~ . , data = train_set_r, method = "rf", trControl = control, preProcess = "nzv", tuneLength = 10)
# Print the model details
print(rf_model_cv_fs)
# Feature importance
importance_scores_r <- varImp(rf_model_cv_fs)$importance
# Set a threshold for feature selection
threshold <- 5  # Adjust this threshold as needed
# Select features based on importance scores
selected_features_r <- rownames(importance_scores_r[importance_scores_r$Overall > threshold, , drop = FALSE])
print(selected_features_r)
# Update the training and testing datasets with selected features
train_set_fs_r <- train_set_r[, c("V4", selected_features_r)]
test_set_fs_r <- test_set_r[, c("V4", selected_features_r)]
# Build a random forest model with selected features
rf_model_fs_r <- randomForest(V4 ~ . , data = train_set_fs_r)
# Print the model details
print(rf_model_fs_r)
# Make predictions on the testing set
pred_fs <- predict(rf_model_fs_r, test_set_fs_r)
# Calculate RMSE
err_fs <- sqrt(mean((test_set_fs_r$V4 - pred_fs)^2))
print(paste("Root Mean Squared Error (Feature Selection + CV):", err_fs))
# Feature importance for the selected features
import_fs <- importance(rf_model_fs)
print(import_fs)

#normal random forest
rf_model_r <- randomForest(V4 ~ . , data = train_set_r)
print(rf_model_r)
pred_r_1<- predict(rf_model_r, test_set_r)
err_r_1 <- sqrt(mean((test_set_r$V4 - pred_r_1)^2))
print(paste("Root Mean Squared Error:", err_r_1))
import_r_1 <- importance(rf_model_r)
print(import_r_1) 


#boosting
library(xgboost)
summary(train_set_r)
str(train_set_r)
class(train_set_r$V4)
xgb_mod <- xgboost(data = as.matrix(train_set_r[, -which(names(train_set_r) == "V4")]), 
                     label = train_set_r$V4, nrounds = 100, objective = "reg:squarederror")
pred_r_2 <- predict(xgb_mod, as.matrix(test_set_r[, -which(names(test_set_r) == "V4")]))
err_r_2 <- sqrt(mean((test_set_r$V4 - pred_r_2)^2))
print(paste("Root Mean Squared Error:", err_r_2))
import_r_2 <- xgb.importance(model = xgb_mod)
print(import_r_2)


#linear regression
library(glmnet)
library(dplyr)
library(caret)
corMatrix <- cor(train_set_r[, -which(names(train_set_r) == "V4")])

# Find features highly correlated with the target variable
highCorr <- findCorrelation(corMatrix, cutoff = 0.7)

# Subset the dataset to keep only the selected features based on correlation
train_set_corr <- train_set_r[, highCorr]
test_set_corr <- test_set_r[, highCorr]

# Print summary of the dataset after correlation-based feature selection
summary(train_set_corr)

# Find features with low variance
lowVar <- nearZeroVar(train_set_corr)

# Subset the dataset to remove the features with low variance
train_set_var <- train_set_corr[, -lowVar]
test_set_var <- test_set_corr[, -lowVar]

# Print summary of the dataset after variance-based feature selection
summary(train_set_var)

# Print out number of selected features after both methods
cat("Number of selected features after correlation-based selection:", ncol(train_set_corr), "\n")
cat("Number of selected features after variance-based selection:", ncol(train_set_var), "\n")

#LR Model with correlation 
lm_mod <- lm(V4 ~ ., data = train_set_corr)
# Predictions
pred_r_3 <- predict(lm_mod, newdata = test_set_corr)
# Calculate RMSE
err_r_3 <- sqrt(mean((test_set_r$V4 - pred_r_3)^2))
print(paste("Root Mean Squared Error:", err_r_3))
# Coefficients
coeff3 <- coef(lm_mod)
print(coeff3)
#CV
library(boot)
# 10-fold cross-validation with FS
ctrl <- trainControl(method = "cv", number = 10)
# Fit the linear model using train() function from caret
lm_mod_cv <- train(V4 ~ ., data = train_set_corr, method = "lm", trControl = ctrl)
# Print cross-validation RMSE
print(paste("Cross-Validation Root Mean Squared Error:", lm_mod_cv$results$RMSE))
coeff_cv <- coef(lm_mod_cv)
print(coeff_cv)

#normal linear regression
lm_mod2 <- lm(V4 ~ . , data = train_set_r)
pred_r_32<- predict(lm_mod2, test_set_r)
err_r_32 <- sqrt(mean((test_set_r$V4 - pred_r_32)^2))
print(paste("Root Mean Squared Error:", err_r_32))
coeff32<- coef(lm_mod2)
print(coeff32)

#linear regression with CV but no FS
lm_mod_cv <- train(V4 ~ ., data = train_set_r, method = "lm", trControl = ctrl)
# Print cross-validation RMSE
print(paste("Cross-Validation Root Mean Squared Error:", lm_mod_cv$results$RMSE))
