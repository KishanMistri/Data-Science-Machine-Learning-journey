#### 1. What does it mean by FPR = TPR = 1 of a model.
    - We can’t have FPR=TPR=1 at the same time equal to 1
    
#### 2. What does AUC = 0.5 signifies.
    - If two points of different classes were given the model would correctly separate them with a probability of 0.5
    - It also means that your model is giving baseline model performance. Which is not expected if it has been tuned.
    
#### 3. When should we use log loss, AUC score and F1 score.
    - If the objective of classification is scoring by probability, it is better to use AUC which averages over all 
    possible thresholds. 
    - If the objective of classification just needs toclassify between two possible classes and doesn't require how 
    likely each class is predicted by the model, it is more appropriate to rely on F-score  using a particular threshold.
    - We use log loss when we care about probability deviations from the actual probabilities.
    
#### 4. What performance metric should use to evaluate a model that see a very less no.of positive data points as compared to -ve data points.
    - Recall as it considers false negatives.
    
#### 5. What performance metric does t-sne use to optimize its probabilistic function.
    - T-sne uses KL divergence between probabilities of two distributions
    
#### 6. What happens in laplace smoothing in my smoothing factor ‘α’ is too large.
    - Underfitting
    
#### 7. When to use cosine similarity over euclidean distance.
    - When the data is very high dimensional due to curse of dimensionality.
    
#### 8. What is fit, transform and fit transform in terms of BOW, tf-idf, word2vector.
    - Fit used to learn the vocabulary and creates a dictionary with key as word and Its words count in the documents as value.
    - Transform converts the given data to the representations that were learnt during training.
    - Fit transform does the two steps on the same data simultaneously.
    
#### 9. How do we quantify uncertainty in probability class labels when using KNN model for classifications.
    - After choosing k nearest neighbors we do majority voting and get its probabilities.
    
#### 10. How do we identify whether the distribution of my train and test is similar or not.
    - we can use QQ-plot over test and train data Or we can measure the distribution of classes among train and test data.
    
#### 11. What does it mean by embedding high dimensional data points to a lower dimension ? what are the advantages and disadvantages of it.
    - It means reducing the dimensionality of the data without losing its meaning by preserving
    the nature like its neighborhood .
    - By dimensionality reduction it becomes much easy to visualize and understand data and models perform 
    well on low dimensional data. Time and space complexity will also be reduced.
    
#### 12. What is the crowding problem w.r.t t-sne.
    - In high dimension we have more room, points can have a lot of different neighbors.
    - In 2D a point can have a few neighbors at distance one all far from each other 
    - what happens when we embed in 1D?
    - This is the ”crowding problem” - we don’t have enough room to accommodate all neighbors.
    - This is one of the biggest problems with SNE.
    - [t-SNE](https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a) solution: 
          Change the Gaussian in Q to a heavy tailed distribution.
    - if Q changes slower, we have more ”wiggle room” to place points at.
    - The next part of t-SNE is to create low-dimensional space with the same number of points as in the 
    original space. Points should be spread randomly on a new space. The goal of this algorithm is to find
    similar probability distribution in low-dimensional space. The most obvious choice for new distribution 
    would be to use Gaussian again. That’s not the best idea, unfortunately. One of the properties of Gaussian
    is that it has a “short tail” and because of that it creates a crowding problem. To solve that we’re 
    going to use Student t-distribution with a single degree of freedom. More of how this distribution was 
    selected and why Gaussian is not the best idea you can find in the [paper](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf   ). I decided not to spend much time 
    on it and allow you to read this article within a reasonable time. 
    
#### 13. What is the need of using log probabilities instead of normal probabilities in naive bayes.
    - Since in original formulation of naïve bayes it has lot of product terms. So, it can lead to 0 values easily. 
    - To overcome this, we use log of probabilities which converts product terms to sum terms.
    
#### 14. What do you mean by hard margin SVM ?
    - n hard margin we only do optimization for increase the margin with the points to be correctly classified.
    - It is under a assumption that there are no misclassified point out of bound of Support Vectors.
   
![SMV](https://user-images.githubusercontent.com/20341930/168136050-d5f11644-2013-4dcd-abb5-e43e4bac66bb.png)

    
#### 15. What is kernel function in svm ?
    - Kernel function in SVM is used to find the similarities of the datapoints by converting them into higher 
    dimensional data where similarities cannot be calculated easily at lower space.
    
#### 16. Why do we call an svm a maximum margin classifier ?
    - In SVM we try to find two parallel hyperplanes which classifies the data which gives high distance 
    difference between them.
    
#### 17. Is svm affected by outliers ?
    - Hard margin SVM will be greatly affected by outliers but soft margin SVM resolves this for some extent 
    by fixing a constraint on misclassification rate.
    
#### 18. Why Locality Sensitive Hashing is not right always?
    - Locality sensitive hashing is method to find similar points that are In the same locality or Neighborhood 
    this is done by drawing multiple hyperplanes and calculating the sides of each point w.r.t each plane thus 
    we treat the points which having same sides as similar.
    - If the 2 points are at the smallest distances, but due to randomized plane it will fall under different buckets so it will not be 100% accurate.
    - Its a probabilistic Algorithm so it won’t give you the right answer but with the highest probability.
    
#### 19. What is sigmoid function? What is its range ?
    - sigmoid function is a function which has a real value for every real value. 
    - Its range is 0 to 1.
    
#### 20. Instead of sigmoid function can we use any other function in LR?
    - We can use any function that can normalize the data and its derivative exists.
    - Example: 𝑍 /(1+|𝑧|)
    
#### 21. Why is accuracy not a good measure for classification problem ?
    - Probability Score of query point belonging to certain class matters as it can makes us confident in 
    decision of classification.
    - This is like saying prob of query point being part of Class A with score of 0.51 and 0.99 are same, Which is wrong.
    
#### 22. How to deal with multiclass classification problem using logistic regression ?
    - We train multiple classifiers each classifies a specific class by using One vs Rest approach.
    
#### 23. Can linear regression be used for classification purpose ?
    - No, because linear regression predicted values are continuous while classification requires probability score.
    - We can do workaround on top of linear regression to make it for classification.
    -Additional reading: https://jinglescode.github.io/2019/05/07/why-linear-regression-is-not-suitable-for-classification/
    
#### 24. What is locality sensitive hashing ?
    - Locality sensitive hashing is method to find similar points that are In the same locality or Neighborhood
    this is done by drawing multiple hyperplanes and calculating the sides of each point w.r.t each plane thus 
    we treat the points which having same sides as similar.
    
#### 25. What is the use of ROC curve ?
    - ROC curves are frequently used to show in a graphical way the connection/trade-off between clinical 
    sensitivity and specificity for every possible cut-off for a test or a combination of tests. 
   - In addition, the area under the ROC curve gives an idea about the benefit of using the test(s) in question.
    
#### 26. When EDA should be performed, before or after splitting data? Why ?
    - we should perform EDA before splitting because we need to understand the whole nature of data which the 
    model would work on. 
    - If EDA was performed on train and the model works better on train and may work worse on test as the model
    is trained on data which couldn’t be prepared for future unseen data.
    
#### 27. How k-nn++ is different from k-means clustering ?
    - In K-means we randomly initialize the centroids at the start. So, model can vary with Different 
    initializations.
    - In K-means++ we choose a better of initializing centroids by following some strategies to avoid 
    problems of initialization.
    
#### 28. Where ensemble techniques might be useful ?
    - Ensembles are used to achieve better predictive performance on a predictive modeling problem than 
    a single predictive model. The way this is achieved can be understood as the model reducing the variance 
    component of the prediction error by adding bias (i.e., in the context of the bias-variance trade-off)
    
#### 29. What is feature forward selection ?
    - In feature forward selection we train a model with one feature at a time and choose the best feature which 
    gives the better performance. 
    - Now we use the best feature and extra one feature one at a time and get the best pair. 
    - This process is repeated until the required most important features are obtained.
    
#### 30. What is feature backward selection ?
    - Feature backward selection is process of selecting important features from that data, where initially 
    we train model on all the features and remove the worse features one by one by until required features 
    are obtained observing the loss or metric.
    
#### 31. What is type 1 & type 2 error ?
    - A type I error is the rejection of a true null hypothesis (also known as a "false positive" finding or conclusion; 
    example: "an innocent person is convicted"), 
    - A type II error is the non-rejection of a false null hypothesis (also known as a "false negative")
    
#### 32. What is multicollinearity ?
    - Multicollinearity occurs when two or more independent variables are highly correlated with one another in a 
    regression model. 
    - This means that an independent variable can be predicted from another independent variable in a regression model.
    
#### 33. How is eigenvector different from other general vectors ?
    - Eigen vector is a vector that signifies most of the variance along a feature
    
#### 34. What is eigenvalue & eigenvectors ?
    - Eigen values gives the amount of variance explained along a vector or feature.
    
#### 35. What is A/B testing?
    - A/B testing is used to know the performance of a model after deployment.
    EX: If a model is retrained and need to be tested. We allow only some part of the traffic through the retrained 
    model and check the performance while compared to the another model. If the new model gives good results, 
    we redirect entire traffic to our New model.
    
#### 36. How to split data which has temporal nature.
    - We should use time-based splitting.
    - EX: If we have data of 10 data we use first 9 days data as train and 10 th day data as test.
    
#### 37. What is response encoding of categorical features ?
    - It is method for encoding categorical features. We will calculate the probability of a class with a given 
    category out of all the points having the given category.
    - Response encoding gives the probability for each class.
    
#### 38. What is the binning of continuous random variables?
    - Data binning (also called Discrete binning or bucketing) is a data pre-processing technique used to reduce 
    the effects of minor observation errors. The original data values which fall into a given small interval, a bin, 
    are replaced by a value representative of that interval, often the central value.
    
#### 39. Regularization parameter in dual form of SVM ?
    - The regularization parameter (lambda) serves as a degree of importance that is given to misclassifications. 
    - SVM pose a quadratic optimization problem that looks for maximizing the margin between both classes and 
    minimizing the amount of misclassifications. 
    -Read full answer here: https://datascience.stackexchange.com/questions/4943/intuition-for-the-regularization-parameter-in-svm
    
#### 40. What is the difference between sigmoid and softmax ?
    - Sigmoid is used if the classes are independent of each other. each class can have any probability value
    without having the constraint of sum of probabilities across all classes to be one. Thus, we can use 
    sigmoid for multi class classification.
    - Softmax is used if the chances of predicting one class effects another. The sum of probabilties of 
    each class should sum to 1 .So, we only have one class prediction at a time.
    
#### 41. For a binary classification which among the following cannot be the last layer ?
```
1. sigmoid(1)
2. sigmoid(2)
3. softmax(1)
4. softmax(2)
```
    - sigmoid(1) [Why?]
    
#### 42. What is P-value in hypothesis testing ?
    - The P value, or calculated probability, is the probability of finding the observed, or more extreme, 
    results when the null hypothesis (H 0) of a study question is true – the definition of 'extreme' depends
    on how the hypothesis is being tested.
    
#### 43. How to check if a particular sample follows a distribution or not ?
    - Use Q-Q plot or K-S test
    
#### 44. What is the difference between covariance and correlation ?
    - Dup
    
#### 45. On what basis would you choose agglomerative clustering over k means clustering and vice versa ?What is the metric that we use to evaluate unsupervised models.
    - If we have similarity matrix, we can use agglomerative clustering.
    - In high dimensional data time complexity of agglomerative clustering is high so we use k-means then.
    
#### 46. What is the difference between model parameters and hyper parameters ?
    - Model parameter are used to define the structural and variations of model.
    - Hyperparameter are used to tune the model performances based on dataset/problem.
    
#### 47. Number of parameters in LSTM is 4m(m+n+1). How many number of parameters do we have in GRU ?
    - 3m(mn+n+1)
    
#### 48. What is box cox transform? When can it be used ?
    - The Box-Cox transformation is a generalized “power transformation” that transforms data to make
    the distribution more normal/Guassian.
    - For example, when its lambda parameter is 0, it’s equivalent to the log-transformation.
    - It’s used to stabilize the variance (eliminate heteroskedasticity) and normalize the distribution.
    
![Box-Cox Transformation equation](https://user-images.githubusercontent.com/20341930/168139153-b4105817-0228-4363-88bd-2beb284c8b4c.png)

    -If λ=0 then X is log-normal
    
#### 49. In what format should the data be sent to embedding layer?
    - Numeric values to be sent to embedding layers
    
#### 50. What is the metric that we use to evaluate unsupervised models?
    - Dunn Index
    
---

#### Question_Template
#### X. Question
    - Answer
