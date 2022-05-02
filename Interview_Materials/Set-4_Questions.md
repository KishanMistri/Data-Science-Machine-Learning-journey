#### 1. Why we need Calibration ?
    - Answer
    
#### 2. What is MAP ? (mean average precision)
    - Answer
    
#### 3. Why do we need gated mechanism in LSTM ?
    - Answer
    
#### 4. What is stratified sampling ? Explain.
    - Problem with random sampling is that Random sampling does not provide the distribution of population of the whole data. 
    So algorithms performing on random sampling provide different results on the test data.
    
    - It is done by dividing the population into subgroups or into strata, and the right number of instances 
    are sampled from each stratum to guarantee that the test set is representative of the entire population.
    
    - Stratified sampling is different from simple random sampling, which involves the random selection of data from 
    the entire population so that each possible sample is equally likely to occur. 
    
    - A random sample is taken from each stratum in direct proportion to the size of the stratum compared to the population, 
    so each possible sample is equally likely to occur.
    
#### 5. How do you compare two distributions ?
    1. QQ Plot
    2. KS Test plot 
    
#### 6. What will happen to train time of K means of data is very high dimension.
    -  When the dimension increases it will have problem with the curse of dimensionality. As the number of dimensions tend to infinity 
    the distance between any two points in the dataset converges. This means the maximum distance and minimum distance between any two 
    points of your dataset will be the same.

    - This is a big problem when you are using the euclidean distance in K-Means.

    - One possible solution if you want to still use K-Means is to change the distance metric you use. I've used spherical K-Means,
    based on the cosine distance with thousands of features without any problems. It is a method that can be used to extract features 
    from images and has results comparable to Deep Learning methods. How about that for an Over-hyped algorithm?

    - So if you have dimensions in the order of thousands or so try K-Means or Spherical K-Means and you will probably have a good solution. 
    If you have many millions of dimensions then you need a subspace clustering algorithm or a dimensionality reduction technique before clustering.
    
#### 7. If you have 10mill records with 100dimension each for a clustering task. Which algorithm will you try first and why ?
    1. K-Means handles well with large data because it has linear time complexity of O(n)
    2. Hierarchical clustering is of order O(n2logn)
    
#### 8. What is matrix Factorization? Explain with an Example.
    - Matrix factorization is a class of collaborative filtering algorithms used in recommender systems. Matrix factorization 
    algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices.
    
#### 9. Which algorithm will give high time complexity if you have 10million records for a clustering task.
    -  Hierarchical clustering gives the high time complexity of order O(n2logn)
    
#### 10. Difference between GD and SGD.
    - Both algorithms are methods for finding a set of parameters that minimize a loss function by evaluating 
    parameters against data and then making adjustments.
    
    - In standard gradient descent, you’ll evaluate all training samples for each set of parameters. This is akin 
    to taking big, slow steps toward the solution.
    
    - In stochastic gradient descent, you’ll evaluate only 1 training sample for the set of parameters before updating 
    them. This is akin to taking small, quick steps toward the solution.
    
#### 11. Which one will you choose GD or SGD? Why ?
    - GD theoretically minimizes the error function better than SGD. However, SGD converges much faster 
    once the dataset becomes large.
    
    - That means GD is preferable for small datasets while SGD is preferable for larger ones.

    - In practice, however, SGD is used for most applications because it minimizes the error function well enough 
    while being much faster and more memory efficient for large datasets.
    
#### 12. Why do we need repetitive training of a model ? 
    - As time goes by, the metrics for accuracy/errors increases as the new data is shifting little bit than the original data we have trained our model with. So to continue to get the expected accuracy, we'll have to train the model with more recent data.
    
    - How often retraining model is always depends on change in data, the point of acceptable performance metric and how often the time to get a new model.
    
#### 13. How do you evaluate the model after productionization ?
    - We can monitor metric that was used to train and evaluate the model.
    - We can create plot of predicted and actual values to see the model prediction is not deviated from margin for production live data.
    - The distribution of production data's predicted class with actual.
    
#### 14. What is need for laplace smoothing in N.B
    - Answer
    
#### 15. Explain Gini impurity.
    - Answer
    
#### 16. Explain entropy?
    - Answer
    
#### 17. How to do multi-class classification with random forest ?
    - Answer
    
#### 18. What is need for CV ?
    - Answer
    
#### 19. What is k-fold cross validation ?
    - Answer
    
#### 20. How do you to CV for a test classification problem using random search.
    - Answer
    
#### 21. Assume We have very high dimension data. Which model will you try and which model will be better in a classification problem.
    - Answer
    
#### 22. What is AUC?
    - Answer
    
#### 23. Tell me one business case where recall is more important than precision. 
    - Answer
    
#### 24. Tell me one business case where precision is more important.
    - Answer
    
#### 25. Can we use accuracy for very much imbalance data? If yes/no , why ?
    - No, you should alway avoid using accuracy where the dataset is imbalance. 

    - Because accuracy will give more generalized information of majority class only. 
    
    It wil dominate and minority class might be completely misclassified though the accuracy won’t have much impact on it.
    
#### 26. Difference between micro average F1 and macro average F1 for a 3 class classification. 
    - Answer
    
#### 27. Difference between AUC and accuracy ?
    - Answer
    
#### 28. How do we calculate AUC for a multiclass classification.
    - Answer
    
#### 29. Test the complexity of Kernel sum ?
    - Answer
    
#### 30. Can we use TSNE for dimensionality reduction i.e convest the data n to d dimension.
    - Answer
    
#### 31. What is pearson correlation coefficient ?
    - Answer
    
#### 32. Training time complexity of naive bayes ?
    - Answer
    
#### 33. 83. Numbers of tunable parameters in maxpooling layer ?
```
    1. (100,50) -> Embeddylayer (36) -> output shape ?
```
    - Answer
    
#### 34. Number of tunable parameters in embedding layer (36, vocab size = 75)
    - Answer
    
#### 35. Relation between KNN and kernel sum ?
    - Answer
    
#### 36. 86. Which is faster
```
    1. SVC(C=1). Fit(x,y)
    2. SGD(Log=hinge).fit(x,y)
```
    - Answer
    
#### 37. Explain about KS test ?
    - Answer
    
#### 38. What is KL divergence ?
    - Answer
    
#### 39. How QQ plot works ?
    - Answer
    
#### 40. What is the need of confidence interval ?
    - Answer
    
#### 41. How do you find the out outliers in the given data set ?
    - Answer
    
#### 42. Can you name a few sorting algorithms and their complexity ?
    - Answer
    
#### 43. What is the time complexity of ”a in list ( )” ?
    - Answer
    
#### 44. What is the time complexity of “a in set ( ) “?
    - Answer
    
#### 45. What is percentile ?
    - Answer
    
#### 46. What is IQR ?
    - Answer
    
#### 47. How do you calculate the length of the string that is available in the data frame column ?
    - Answer
    
#### 48. Can you explain the dict.get() function ?
    - Answer
    
#### 49. Is list is hash table ?
    - Answer
    
#### 50. Is tuple is hash table ?
    - Answer
    
---

#### Question_Template
#### X. Question
    - Answer
