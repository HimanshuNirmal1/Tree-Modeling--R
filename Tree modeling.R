# Lab 5
# Author: Himanshu Nirmal
# Part 1

# This problem involves the OJ data set, which is part of the ISLR package.
# 1. Create a training set containing a random sample of 800 observations, and a test set containing the remaining 
# observations.

library(tree)
library(ISLR)
set.seed(67)
dataframe = sample(1:nrow(OJ), 800)
trains = OJ[dataframe,]
tests = OJ[-dataframe,]

# 2. Fit a tree to the training data, with Purchase as the response and the other
# variables except for Buy as predictors. Use the summary() function to produce
# summary statistics about the tree, and describe the results obtained. What is the
# training error rate? How many terminal nodes does the tree have?

tree1 = tree(Purchase ~., trains)
summary(tree1)

# Misclassification error rate: 0.1662 = 133 / 800 
# Number of terminal nodes:  7

# 3. Type in the name of the tree object in order to get a detailed text output. Pick one
# of the terminal nodes, and interpret the information displayed.

tree1

# Let us consider node number 7: LoyalCH with a split criteria of LoyalCH > 0.764572
# Wherein 97.82% are part of CH while only upto 2.18% are the value of MM.

# 4. Create a plot of the tree, and interpret the results.

plot(tree1)
text(tree1, pretty = 0)

#LoyalCH is an important predictor. The topmost 3 nodes are all LoyalCH

# 5. Predict the response on the test data, and produce a confusion matrix comparing
# the test labels to the predicted test labels. What is the test error rate?

treepred = predict(tree1, tests, type = "class")
table(treepred, tests$Purchase)
mean(treepred != tests$Purchase)

# Test Error rate is 0.1629 or around 16.3%

# 6. Apply the cv.tree() function to the training set in order to determine the optimal tree size.

cvtree = cv.tree(tree1, FUN = prune.misclass)
cvtree

# 7. Produce a plot with tree size on the x -axis and cross-validated classification error
# rate on the y -axis.


plot(cvtree$size, cvtree$dev, type = "b", xlab = "Tree size", ylab = "CV Error")


# 8. Which tree size corresponds to the lowest cross-validated classification error rate?

# tree 7 has the lowest cross-validated classification error rate!

# 9. Produce a pruned tree corresponding to the optimal tree size obtained using crossvalidation.
# If cross-validation does not lead to selection of a pruned tree, then
# create a pruned tree with five terminal nodes.

prunetree = prune.misclass(tree1, best = 2)
plot(prunetree)
text(prunetree, pretty = 0)

# 10. Compare the training error rates between the pruned and un-pruned trees. Which is higher? 
summary(tree1)
summary(prunetree)

# Misclassification error rate un-pruned: 0.1662 = 133 / 80
# Misclassification error rate pruned: 0.195 = 156 / 800

# Pruned tree has a higher error rate

# 11. Compare the test error rates between the pruned and un-pruned trees. Which is higher?

prune1 = predict(prunetree, tests, type = "class")
table(prune1, tests$Purchase)
mean(predict(prunetree, tests, type='class')!=tests$Purchase)

# Test Error rate is 0.2148 or around 21.48% which is higher than un-pruned (16.3%)


## Part II  
# We will use Carseats data and seek to predict Sales using regression trees and related approaches, 
# treating the response as a quantitative variable. 

# 1. Split the data set into a training set and a test set. 

set.seed(87)
df = sample(1:nrow(Carseats), nrow(Carseats) / 2)
trains = Carseats[df, ]
tests = Carseats[-df, ]

# 2. Fit a regression tree to the training set. Plot the tree, and interpret the results. 
# What test MSE do you obtain? 

tree2 = tree(Sales ~ ., data = trains)
summary(tree2)
plot(tree2)
text(tree2, pretty = 0)
pred2 = predict(tree2, newdata = tests)
mean((pred2 - tests$Sales)^2)

# Test MSE obtained is 5

# 3. Use cross-validation in order to determine the optimal level of tree complexity.
# Does pruning the tree improve the test MSE? 

cvcar = cv.tree(tree2)
plot(cvcar$size, cvcar$dev, type = "b")
treem = which.min(cvcar$dev)
points(treem, cvcar$dev[treem], col = "Blue", cex = 2, pch = 20)

prunecar = prune.tree(tree2, best = 8)
plot(prunecar)
text(prunecar, pretty = 0)
pred3 = predict(prunecar, newdata = tests)
mean((pred3 - tests$Sales)^2)


# Test MSE obtained is 5.1 which is slightly higher than the un-pruned tree

# 4. Use the bagging approach in order to analyze this data. What test MSE do you obtain? 
# Use the importance() function to determine which variables are most important. 

require(randomForest)
bagcars = randomForest(Sales ~ ., data = trains, mtry = 10, ntree = 500, importance = TRUE)
bagpredcar = predict(bagcars, newdata = tests)
mean((bagpredcar - tests$Sales)^2)
importance(bagcars)

# Bag Test MSE is lowest with 2.9 Importance() tells us that ShelveLoc(64.4%) is most important variable followed by
# Price(56.62%) and then CompPrice(22.63%)

# 5. Use random forests to analyze this data. What test MSE do you obtain? 
# Use the importance() function to determine which variables are most important.
# Describe the effect of m, the number of variables considered at each split, on the error rate obtained. 

randomcars = randomForest(Sales ~ ., data = trains, mtry = 3, ntree = 500, importance = TRUE)
rfpredcar = predict(randomcars, newdata = tests)
mean((rfpredcar  - tests$Sales)^2)
importance(randomcars)

# RandomForests Test MSE is higher with a value of 3.15 since m = p^1/2 i.e (root of p) Importance() shows us that
# ShelveLoc again is most important predictor with 40.11% followed by Price(35.44%) and Advertising(13.16%)







