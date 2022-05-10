#### 1. Why we need Calibration ?
    - After training a model, the model may or may not give the exact probabilities as the model may have many 
    assumptions while training. So, to obtain exact probability values we use calibration.
    
#### 2. What is MAP ? (mean average precision)
    - MAP is calculated in case of multi-class classification problems where Average Precision is calculated for 
    all classes and mean of average precision is calculated.
    
#### 3. Why do we need gated mechanism in LSTM ?
    - In LSTM’s long-term dependencies of input preserved by the mechanism of gates in LSTM which enables model to
    work on long sequences. Forgot gate enables to change the amount of input to be transferred to another cell and so on.
    
#### 4. What is stratified sampling ? Explain.
    - Problem with random sampling is that Random sampling does not provide the distribution of population of the whole data. 
    So algorithms performing on random sampling provide different results on the test data.
    
    - It is done by dividing the population into subgroups or into strata, and the right number of instances 
    are sampled from each stratum to guarantee that the test set is representative of the entire population.
    
    - Stratified sampling is different from simple random sampling, which involves the random selection of data from 
    the entire population so that each possible sample is equally likely to occur. 
    
    - A random sample is taken from each stratum in direct proportion to the size of the stratum compared to the population, 
    so each possible sample is equally likely to occur.
    
    - In stratified sampling each sample gets equally shared class data points after dividing Into smaller groups.
    
#### 5. How do you compare two distributions ?
    1. Kullback-Leibler Divergence

![image](https://user-images.githubusercontent.com/20341930/167595286-e46e1bee-6b7d-4c63-8465-dd26181c3f11.png)

    2. KS Test plot 
  
    3. QQ Plot
    
#### 6. What will happen to train time of K means of data is very high dimension.
    -  When the dimension increases it will have problem with the curse of dimensionality. As the number of dimensions 
    tend to infinity the distance between any two points in the dataset converges. This means the maximum distance and 
    minimum distance between any two points of your dataset will be the same.

    - This is a big problem when you are using the euclidean distance in K-Means.

    - One possible solution if you want to still use K-Means is to change the distance metric you use. I've used spherical 
    K-Means,based on the cosine distance with thousands of features without any problems. It is a method that can be 
    used to extract features from images and has results comparable to Deep Learning methods. How about that for 
    an Over-hyped algorithm?

    - So if you have dimensions in the order of thousands or so try K-Means or Spherical K-Means and you will probably
    have a good solution. If you have many millions of dimensions then you need a subspace clustering algorithm or 
    a dimensionality reduction technique before clustering.
    
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
    - As time goes by, the metrics for accuracy/errors increases as the new data is shifting little bit than the original 
    data we have trained our model with. So to continue to get the expected accuracy, we'll have to train the model with 
    more recent data.
    
    - How often retraining model is always depends on change in data, the point of acceptable performance metric and how 
    often the time to get a new model.
    
#### 13. How do you evaluate the model after productionization ?
    - We should perform A/B testing.
    - We can monitor metric that was used to train and evaluate the model.
    - We can create plot of predicted and actual values to see the model prediction is not deviated from margin for 
    production live data.
    - The distribution of production data's predicted class with actual.
    
#### 14. What is need for laplace smoothing in N.B
    - alpha in Laplace smoothing will act as regularizer.
    - As alpha increases the likelihood probabilities will have uniform distribution 
    i.e.
        P(0)=0.5 and P(1)=0.5 thus the prediction will be biased towards larger class.
        As alpha decreases even with the small change in the data will affect the probability
        Thus, it leads to overfitting
    
#### 15. Explain Gini impurity.
    - Gini Impurity is similar to Entropy which explains the randomness of the data.
    - Gini impurity is a function that determines how well a decision tree was split. Basically, it helps us to 
    determine which splitter is best so that we can build a pure decision tree. Gini impurity ranges values from 0 to 0.5. 
    
![GINI Impurity](https://user-images.githubusercontent.com/20341930/167596679-7ab91543-94fd-45b8-95a7-d37058a4f920.png)

    
#### 16. Explain entropy?
    - Entropy, as it relates to machine learning, is a measure of the randomness in the information being processed. 
    - The higher the entropy, the harder it is to draw any conclusions from that information.
        If the Entropy is 0, we can say that the split is done well and 
        if Entropy is 1, we have the Equally distributed classes in the split.

#### 17. How to do multi-class classification with random forest ?
    - Answer
    
#### 18. What is need for CV ?
    - While doing hyperparameter tuning we can’t use test data as it leads to data leakage. So, we use cross validate data 
    for hyperparameter tuning.
    
#### 19. What is k-fold cross validation ?
    - In training process for hyperparameter tuning we make use of test data but that leads  to leakage of data. 
    - To use our training data itself for doing hyperparameter effectively we do k-fold cross validation. 
    - In which we divide train data in k parts and at each iteration we use k-1 parts for training and other 
    part for hyperparameter thus, we get k iterations enabling no data loss while training.
    
#### 20. How you do CV for a test classification problem using random search.
    -  In random search cv we try out all the combinations of hyperparameters from a list of values and try to pick 
    the combination which gives best score. 
    
#### 21. Assume We have very high dimension data. Which model will you try and which model will be better in a classification problem.
    - Naïve Bayes which can do well on high dimensional data compared to many other Models, this can be used as 
    a benchmark for training.
    
#### 22. What is AUC?
    - AUC is short form of Area under curve. 
    - ROC curve is drawn by plotting FPR vs TPR with each class probability of a datapoint treated as threshold.
    Range of AUC is [0,1] 0-is worst, 1- is best, random model can get AUC of 0.5.
    - AUC tells that if two points of different class labels given how likely the model would correctly classify them.
    
![AUC-ROC Curve](https://user-images.githubusercontent.com/20341930/167599426-fa9f46b8-d16f-4cf6-ae88-504aeacbea9a.png)

    
#### 23. Tell me one business case where recall is more important than precision. 
    - In Medical treatments we can afford to miss any person in giving vaccine who are tested positive. And we can leave 
    the persons without vaccinating who are tested negative.
    
#### 24. Tell me one business case where precision is more important.
    - In medical treatments we should not give treatment for wrong person who are tested  Negative which can lead to side effects.
    
    - Precision is more important than recall when you would like to have less False Positives in trade off to have 
    more False Negatives. Meaning, getting a False Positive is very costly, and a False Negative is not as much.
    
    - In a zombie apocalypse, of course you would try to accept as many as healthy people you can into your safe zone, 
    but you really don’t want to mistakenly pass a zombie into the safe zone. So, if your method causes some of the 
    healthy people mistakenly not to get into the safe zone, then so be it.

    - When you let go 100 culprits your recall is low. But if you punish someone, you are sure that you are punishing 
    only a criminal - precision is high.
    
#### 25. Can we use accuracy for very much imbalance data? If yes/no , why ?
    - No, you should alway avoid using accuracy where the dataset is imbalance. 

    - Because accuracy will give more generalized information of majority class only. 
    
    - Example: we have 100 datapoints of 80 +ve and 20 -ve, even if the model predicts all points
    to be +ve we get an accuracy of 80%
    
    It wil dominate and minority class might be completely misclassified though the accuracy won’t have much impact on it.
    
#### 26. Difference between micro average F1 and macro average F1 for a 3 class classification. 
    - In micro average F1-score we care for all true positive and true negatives of each Class and take all values
    into accounts.
    - This deal with imbalanced data well
    - In macro average F1-score we don’t care for individual classes we do the average of Precision and recall calculated 
    on individual classes. This doesn’t account for class Imabalance.
    
#### 27. Difference between AUC and accuracy ?
    - Accuracy tells with how much percentage the model correctly classifies a point.
    - AUC tells that with how much percentage the model correctly classifies two or more points correctly.
    
#### 28. How do we calculate AUC for a multiclass classification.
    - We need to one vs all approach and calculate AUC for each class.
    
#### 29. Test the complexity of Kernel SVM ?
    - Best Cases:
        Right Kernel for given data set
    
    - Worst Case:
        N is large (O(N^2))
        K (Number of support Vectors) is large O(k*d)
    
#### 30. Can we use TSNE for dimensionality reduction i.e convert the data n to d dimension.
    - No
    - The main reason that t-SNE is **not** used in classification models is that it does **not** learn a function from 
    the original space to the new (lower) dimensional one. 
    - As such, when we would try to use our classifier on new / unseen data we will not be able to map / preprocess these
    new data according to the previous t-SNE results.
    - t-SNE tries to find new distribution of data in lower space such that both the distributions are very similar. 
    This is achieved by KL divergence
    
#### 31. What is pearson correlation coefficient ?
    - person correlation coefficient is a method by which we can obtain the variability of two random variables. 
    - Unlike co-variance it gives exact value by which the variables are related to each other.
    - It measures the linear correlation between two variables.
    - Its range is -1 to 1.
    
#### 32. Training time complexity of naive bayes ?
    - Runtime Space required very less.  
        Time:: O(Records * Dimension * Classes)  
        Space: O(Records * Dimension)
    
#### 33. 83. Numbers of tunable parameters in maxpooling layer ?
```
    1. (100,50) -> Embeddylayer (36) -> output shape ?
```
    - 0
    
#### 34. Number of tunable parameters in embedding layer (36, vocab size = 75)
    - 75*36 = 2700
    
#### 35. Relation between KNN and kernel sum ?
    - In KNN we use similarities of points based on distances similarly we use similarities Computed by kernel in kernel SVM.
    - In RBF kernel σ is equivalent to K in KNN, i.e., as σ increases we consider points with distances of increased range 
    similarly in KNN if K is increases, we take more points having more distance ranges.
    
#### 36. 86. Which is faster
```
    1. SVC(C=1). Fit(x,y)
    2. SGD(Log=hinge).fit(x,y)
```
    - SGD with hinge loss will be faster because it doesn’t have any constraints like
    - SVC with soft margin has Constraint = yi(wtxi+b)>=1-ξi
    
#### 37. Explain about KS test ?
    - KS Test tells us that whether two distributions are same or not.
    
![KS Test Graph](https://user-images.githubusercontent.com/20341930/167602824-f9f6256d-0203-4420-a294-4e845b81ab37.png)

    
#### 38. What is KL divergence ?
    - Answer
    
#### 39. How QQ plot works ?
    - QQ plot is used to know the distribution of unknown data.
    - 1. The values are sorted, and quantiles is computed.
    - 2. Now we take the values from a known distribution and computed percentiles in the same way.
    - 3. Now we plot this percentile values if the plot is a straight line, we can say that both the Distributions are same. 
    Thus, we can know the distribution of unknown data.
    
#### 40. What is the need of confidence interval ?
    - Confidence intervals show us the likely range of values of our population mean. When we calculate the mean we just 
    have one estimate of our metric; confidence intervals give us richer data and show the likely values of 
    the true population mean.
    
#### 41. How do you find the out outliers in the given data set ?
    - Using Box plot
    - Plotting the pdf of the data.
    
#### 42. Can you name a few sorting algorithms and their complexity ?
    - 
![Time Complexity of Algorithms](https://user-images.githubusercontent.com/20341930/167603586-24c8fd2b-74a4-4764-a759-ed6c509b185d.png)

    
#### 43. What is the time complexity of ”a in list ( )” ?
    - O(N) => Need to traverse whole list
    
#### 44. What is the time complexity of “a in set ( ) “?
    - O(1) => Key val pair.
    
#### 45. What is percentile ?
    - percentile is a value that says what is the value of data present at a specific percentage when the data is sorted.
    
#### 46. What is IQR ?
    - IQR tells what the values are present in 25th and 75th percentile range of data.
      IQR=Q3-Q1
    
#### 47. How do you calculate the length of the string that is available in the data frame column ?
```
len(df.iloc[index][column])
```

#### 48. Can you explain the dict.get() function ?
    - It takes a key as argument and returns its value if the key is not present returns None.
    
#### 49. Is list is hash table ?
    - No
    
#### 50. Is tuple is hash table ?
    - No
    
---

#### Question_Template
#### X. Question
    - Answer
