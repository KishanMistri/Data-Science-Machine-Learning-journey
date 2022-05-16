#### 1. Which central tendency method is used if there exist any outliers?
    - median are not affected by outliers.
    
#### 2. Explain, What is Central limit theorem?
    - The Central Limit Theorem states that the sampling distribution of the sample means approaches a normal 
    distribution as the sample size gets larger — no matter what the shape of the population distribution. 
    - This fact holds especially true for sample sizes over 30.

    - All this is saying is that as you take more samples, especially large ones, your graph of the sample means
    will look more like a normal distribution.
    
    
#### 3. What is Chi-Square test?
    - There are two types of chi-square tests. Both use the chi-square statistic and distribution 
    for different purposes:
    
    1. A chi-square goodness of fit test determines if sample data matches a population. 
    
    2. A chi-square test for independence compares two variables in a contingency table to see if they are
    related. In a more general sense, it tests to see whether distributions of categorical variables differ
    from each another.
    
#### 4. What is A/B testing?
    - A/B testing is a basic randomized control experiment. It is a way to compare the two versions of a 
    variable to find out which performs better in a controlled environment.
    
#### 5. Tell us the difference between Z and t distribution (Linked to A/B testing)?
    - T-test
        - Why? - To compare MEANS of 2 population to get a conclusion on Hypothesis testing.
        - When? - When sample/population size is small.
        
    - Z-test
        - Z = Standard Normal Variate
        - 2 samples test [For K samples → ANOVA test]
        - Why? - A z-test is a statistical test **to determine whether two population means are different 
        when the variances are known and the sample size is large**. A z-test is a hypothesis test in 
        which the z-statistic follows a normal distribution.
        
        - When? - When sample/population size is sufficiently large (~100).
    
#### 6. Tell some outlier treatment methods.
    - Drop
    - Prune with mean, mode , or any other ML regression/classification technique
    
#### 7. What is ANOVA test?
    - We are doing 2 sample test (comparison of 2 samples), but what if I have k samples to test. 
    We can do 2-2 one at a time but not efficient.
    - Assumptions:
        - Each groups populations (recovery time observations) are Gaussian.
        - Each Group’s variance is same.
        - Each observation is independent.
    
#### 8. What is Cross validation?
    - To verify the model performance and fit it properly as it get the generalization of data we create different 
    sets from training and test on different subset of data and fit the model.
    - That way it is able to make more generalize model which works better on unseen real data.
    
#### 9. How will you work in a machine learning project if there is a huge imbalance in the data?
    - Upsampling and downsampling
    - Weighting
    - Synthetically producting minority dataset class to match majority class.
    
#### 10. Tell the formula of sigmoid function.
    - Sigmoid function f(x) = 1/(1+exp^(x))
    
#### 11. Can we use sigmoid function in case of multiple classifications?
    - softmax() will give you the probability distribution which means all output will sum to 1. 
    While, sigmoid() will make sure the output value of neuron is between 0 to 1.
    - So Yes, both work the same way. Softmax is an extension of sigmoid for multi-class classifications problem. 
    Softmax in multiclass logistic regression with K=2 takes the form of sigmoid function
    
#### 12. What is Area under the curve (AUC)?
    - The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between
    classes and is used as a summary of the ROC curve. The higher the AUC, the better the performance of 
    the model at distinguishing between the positive and negative classes.
    
#### 13. Which metric is used to split a node in Decision Tree?
    - Decision tree algorithms use information gain to split a node
    - Gini index and entropy is the criterion for calculating information gain.
    
#### 14. Explain ensemble learning?
    - Using multiple weak learner and making average of those n number of model which gives better performance and randomness.
    
#### 15. What is P value?
    - In statistics, the p-value is the probability of obtaining results at least as extreme as the observed 
    results of a statistical hypothesis test, assuming that the null hypothesis is correct. 
    - The p-value serves as an alternative to rejection points to provide the smallest level of significance 
    at which the null hypothesis would be rejected. 
    - A smaller p-value means that there is stronger evidence in favor of the alternative hypothesis.
    
#### 16. What are histograms?
    - Frequency count of feature in the given dataset.
    
#### 17. Tell us about confidence interval?
    - We can't calculate the population mean accurately due to constrains like it being impossible like of 
    getting all male heights, all votes or unavailable of the dataset, 
    but we want to conclude it in a such a way that it would be probabiliticly accurate.
    
    - We take a sample of observed values and using randomized sampling from observations, a confidence interval which 
    is the mean of your estimate plus and minus the variation in that estimate. 
    
    - This is the range of values you expect your estimate to fall between if you redo your test, 
    within a certain level of confidence.
    
#### 18. What’s the reason for high bias or variance?
    - A model with high variance may represent the data set accurately but could lead to overfitting
    to noisy or otherwise unrepresentative training data. 
    
    - In comparison, a model with high bias may underfit the training data due to a simpler model that
    overlooks regularities in the data.
    
#### 19. Which models are generally high biased or high variances?
    - A linear machine-learning algorithm will exhibit high bias but low variance. 
    - On the other hand, a non-linear algorithm will exhibit low bias but high variance. 
    - Using a linear model with a data set that is non-linear will introduce bias into the model.
    
#### 20. Why do we select validation data other than test data?
    - To make model more generalized before using it on test data.
    
#### 21. What are the differences between linear and logistic regression?
    - Logistic regression is Classification technique
    - Linear regression is regression technique which fits and predicts linearly (on line)
    - They are totally different...
    
#### 22. Why do we take such a complex cost function for logistic regression?
    - Refer this answer in stats exchange: https://stats.stackexchange.com/questions/174364/why-use-different-cost-function-for-linear-and-logistic-regression
    
#### 23. Differentiate between random forest and decision tree?
    - ![image](https://user-images.githubusercontent.com/20341930/168646391-f245747c-fde2-4f42-81c0-5e28ad975d82.png)
    
#### 24. How would you decide when to stop splitting the tree?
    - There are many ways and all of them is listed in this article. [Not to make this page repo for article
    but to create a question and asnwer with max understanding]
    
   - Refer: https://machinewithdata.com/2018/06/24/how-and-when-does-the-decision-tree-stop-splitting/
    
#### 25. What are the measures of central tendency?
    - mean, median, mode, midrange
    
#### 26. What is the requirement of k means algorithm?
    - As this is unsupervised learning, we need to give the number of cluster as a stopping criteria and as a similarity groups.
    
#### 27. Which clustering technique uses combination of clusters?
    - DBSCAN & Agglomerative Clustering algorithms
    
#### 28. Which is the oldest probability distribution?
    - Binomial [1713]
    - Normal [1807]
    
#### 29. What all values can a random variable take?
    - Descrete random variable takes set of values
    - On the other hand, continuous random variable takes countably infinite number of values.
    
#### 30. What are the different types of random variables?
    - Continuous & Descrete
    
#### 31. Describe normality of residuals.
    - It means all the residuals are normally distributed.
    - Use ANOVA test to validate.
    
#### 32. What is T-test used for?
    - Why? - To compare MEANS of 2 population to get a conclusion on Hypothesis testing.
    - When? - When sample/population size is small.
    
#### 33. How do you perform dimensionality reduction?
    - There are n number of techniques to perform dimensionality reduction:
      - Principal Component Analysis.
        Backward Elimination.
        Forward Selection.
        Score comparison.
        Missing Value Ratio.
        Low Variance Filter.
        High Correlation Filter.
        Random Forest.
    
#### 34. What are the assumptions of linear regression algorithm?
    - There are four assumptions associated with a linear regression model:

        1. Linearity: The relationship between X and the mean of Y is linear.
        2. Homoscedasticity: The variance of residual is the same for any value of X.
        3. Independence: Observations are independent of each other.
        4. Normality: For any fixed value of X, Y is normally distributed.
        
    
#### 35. Differentiate between Correlation and covariance.
    - Covariance shows you how the two variables differ
    - whereas Correlation shows you how the two variables are related.
    
#### 36. How to identify & treat outliers and missing values?
    - Drop
    - Prune
    
#### 37. Explain Box and whisker plot.
    - Box is IQR => 25%tile to 75%tile (Can be customize)
    - Whiskers are 1.5 IQR from 25%tile & 75%tile.
    
#### 38. Explain any unsupervised learning algorithm.
    - Kmeans
    - Kmeans++
    - DBSCAN
    - Hierarchical Clustering
    
#### 39. Describe Random Forest.
    - https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/
    
#### 40. What packages in Python can be used for ML? Why do we prefer one over another?
    - Skelarn and other library with specific algorithm implementation.
    - Sklearn is generalize formulation with excellent documentation. however, It also limits some of the
    parameter tweaking so when we need base parameter tweaks then we should be refer source library.
    - This is just one example based on experiment. There is alot of examples as such.
    
#### 41. What are the Evaluation Metric parameters for testing Logistic Regression?
    - RMSE
    
#### 42. NumPy vs Pandas basic difference.
    - Numpy is working on python list where optimization is that python list can contain different data types.
    While Numpy array can only have single datatype which makes it incredible fast.
    
    - Pandas uses Numpy as a lower level implementation.
    
#### 43. Tuple vs Dictionary. Where do we use them?
    - Tuples are immutable data structure.
    - Dictionary is mutable and well structured data structure.
    
#### 44. What is NER(Named Entity Recognition)?
    - Named Entity Recognition can automatically scan entire articles and reveal which are the major people, 
    organizations, and places discussed in them. Knowing the relevant tags for each article help in automatically 
    categorizing the articles in defined hierarchies and enable smooth content discovery.

    - Example: When we read a text, we naturally recognize named entities like people, values, locations, and so on. 
    In the sentence “Mark Zuckerberg is one of the founders of Facebook, a company from the United States” we can 
    identify three types of entities: “Person”: Mark Zuckerberg. “Company”: Facebook.
    
    
#### 45. Can Linear Regression be used for Classification? If Yes, why if No why?
    - There are two things that explain why Linear Regression is not suitable for classification. 
    
    1. The first one is that Linear Regression deals with continuous values whereas classification
    problems mandate discrete values. 
    
    2. The second problem is regarding the shift in threshold value when new data points are added.
    
#### 46. What is Naive Bayes Theorem? Multinomial, Bernoulli, Gaussian Naive Bayes.
    - Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. 
    It is not a single algorithm but a family of algorithms where all of them share a common principle, 
    i.e. every pair of features being classified is independent of each other.
    
#### 47. Differentiate between Over Sampling and Under Sampling.
    - Oversampling: Synthecize minority class to match ratio to majority class [preffered]
    -Undersampling: Selecting subset of majority class which matches minority class.
    
#### 48. what is the different between Over Fitting and Under Fitting.
    - Overfitting:  overly Complex model. Fitted to training set. Not generalized.
    - Underfitting: Simple Model. Not fitting data properly. 
    
#### 49. Differentiate between Gini Index and Entropy.
    - The Gini Index and the Entropy have two main differences: Gini Index has values inside the interval [0, 0.5]
    - The interval of the Entropy is [0, 1]. 
    - Gini is better for selecting features as the range is smaller so concise.
    
#### 50. What are the advantages and disadvantages of PCA?
    - Refer: https://www.i2tutorials.com/what-are-the-pros-and-cons-of-the-pca/
    
---

#### Question_Template
#### X. Question
    - Answer
