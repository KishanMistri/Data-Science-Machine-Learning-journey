#### 1. What is parameter sharing in deep learning?
    - A convolutional neural network learns certain features in images that are useful for classifying the image. 
    Sharing parameters gives the network the ability to look for a given feature everywhere in the image, rather 
    than in just a certain area. This is extremely useful
    - when the object of interest could be anywhere in the image. Relaxing the parameter sharing allows the 
    network to look for a given feature only in a specific area. For example, if your training data is of faces 
    that are centered, you could end up with a network that looks for eyes, nose, and mouth in the center of
    the image, a curve towards the top, and shoulders towards the bottom.
    - It is uncommon to have training data where useful features will usually always be in the same area, so 
    this is not seen often.
    
#### 2. What will be the alpha value for non support vectors.
    - 0 for non-support vectors
    - >=0 for support vectors
    
#### 3. What will be the effect of increasing alpha values in multinomial NB ?
    - It leads to underfitting.
    
![NB with Laplace smoothing with Alpha](https://user-images.githubusercontent.com/20341930/167795869-25a27678-958a-436c-9ef4-768a59525dc4.png)
    
    - Generally alpha =1 is used. If it is increasing then denominator increases very fast and Prob values are low. Which makes less accurate and simple model. Hence Underfitting.
    
#### 4. What is recurrent equation of RNN output function ?

![Eq for RNN output function](https://user-images.githubusercontent.com/20341930/167796496-b5266308-302f-4a7c-b0a0-ce711a6f65af.png)

    
#### 5. What is the minimum and maximum value of tanh ?
    - tanh is hyperbolic tangent function whose value reside between -1 and +1.
    
![image](https://user-images.githubusercontent.com/20341930/165881784-bbba1fe6-e7d4-4204-83a0-fc5fb2c3ab50.png)

    
#### 6. How many thresholds we need to check for a real valued features in DT ?
    - In DT with real values, we need to treat each value as a threshold
    
#### 7. How do you compute the feature importance in DT ?
    - We take the node which gives high information gain as important.
    - Feature importance is calculated as the decrease in node impurity weighted by the probability of reaching that node. 
    The node probability can be calculated by the number of samples that reach the node, divided by the total number of
    samples. The higher the value the more important the feature.
    
#### 8. How do you compute the feature importance in SVM ?
    - By using the weight coefficients.
    - In Kernel it is not possible because we use the datapoints of higher dimensions for Classification. Which is not 
    directly related to current set of features.
    
#### 9. Prove that L1 will given sparsity in the weight vector ? OR Why does L1 Reg creates sparsity?
    - Derivations are main reason to converging pace.

    - In L2, the slope decreases slowly so it will take time to become Zero

    - In L1, the Slope is contant (if p>0  →+1 OR if p<0 → -1) so it will become zero early 
    or with very less iteration than L2.

![https://www.appliedroots.com/images/eif/6285_1641586684.png](https://www.appliedroots.com/images/eif/6285_1641586684.png)

    - L2 regularization tries to bring the value close to zero.

    - L1 regularization tries to bring the value very very close to zero.
    
#### 10. What are L1,L2 regularizers ?
    - When we are using any algorithm, we have loss-function/equation to determind the class. 
    So when we use it we will get the class of query point. Here we have assumed that there are 
    no outliers OR the data is linearly separable. However, in real life exmaples, it will not be the case. 
    So Regularization is the mechanism to penalize the false predictions and prevent overfitting and 
    underfitting (Along with hyperparameters).

    - Optimization problem = Loss_function + Regularization

    - L1 OR Lesso regression: It uses absolute value of magnitude
        regularization = Hyperparameter * |W|
    - L2 or Ridge Regularization:  It uses squared value of magnitude
        regularization = Hyperparameter * |W|^2
    
#### 11. What is elastic net ?
    - We can also use both L1 & L2 regularization jointly as a part of optimization problem, 
    Which is called ELASTIC NET formulation.
    - However there will be 2 hyperparameters.
    - Regularization term = lambda_1 * |Wi|_1) + (lambda_2 * |Wi|_2)
    
#### 12. What are the assumption of NB ?
    - That all the features are conditionally independent. 
    Which is not seen in real world, every feature has some kind of dependency (minimum/very low).
    
#### 13. What are the assumptions of KNN ?
    - KNN uses neighborhood so basic assumption is that the point/data point in the close proximity is the basis of same class classification
    
#### 14. What are the assumptions of linear regression ?
    - Linear regression is an analysis that assesses whether one or more predictor variables explain the dependent 
    (criterion) variable.  The regression has five key assumptions:
        Linear relationship
        Multivariate normality
        No or little multicollinearity
        No auto-correlation
        Homoscedasticity
    A note about sample size.  In Linear regression the sample size rule of thumb is that the regression analysis 
    requires at least 20 cases per independent variable in the analysis.
    
#### 15. Write the optimization equation of linear regression ?
    
![Linear Regression Optimization Equation](https://user-images.githubusercontent.com/20341930/167810834-3ed08514-6b30-4102-b563-0cf33baa7962.png)


#### 16. What is time complexity of building KD tree ?
    - O(n*d*logn)
    
#### 17. What is the time complexity to check if a number is prime or not ?
    - O(sqrt(n))
    
#### 18. Angle between two vectors (2,3,4) (5,7,8).
    - Calculation:

![Q_cosine](https://user-images.githubusercontent.com/20341930/167812412-551756b1-d278-4e39-91b5-f7273ebd6eba.png)

    
#### 19. Angle between the weigh vector of 2x+3y+5=0 and the vector(7,8).
    -  Cos 0 = 0.5
    
#### 20. Distance between (7,9) and the line 7x+4y-120=0.
    - Answer: 4.34
    - ax+by+c=0 => 7x+4y-120=0
    - (x1, y1)  => (7,9)
    => Distance = || a*x1 + b*y1 + c || / sqrt (a^2 + b^2)
    
#### 21. Distance between the lines 4x+5y+15=0, 4x+5y-17=0.
    - two lines are parallel so we get 15-(-17)=32 from intecepts
    
#### 22. 122. Which of this hyperplane will classify these two class points correctly?
```
    1. Positives: A(2,3), B(-3,4) Negatives: C(-5,7), D(-5,-9)
    2. Pi_1: 4x+5y+7=0, pi_2: -3y+3x+9=0
```
    - Pi_1:
        Correctly classifies: A, B, C
        Mis Classify: D
    - Pi_2:
        Correctly classifies: A, C
        Mis Classify: B, D
    
    
#### 23. 123. Which of the vector pairs perpendicular to each other
```
    1. (3,4,5) (-3,-4,5)
    2. (7,4,6) (-4,-7,-12)
```
    - Answer: Cos0 = 90' for vectors  1) (3,4,5) (-3,-4,5)
    
#### 24. How dropout works ?
    - At each iteration dropouts layer makes a neuron active with a probability of dropout rate. This makes the model 
    to avoid overfitting as only some neurons are at each step.
    
![NN layer dropout](https://user-images.githubusercontent.com/20341930/168097390-bdcb23ee-52b3-4d17-8880-3747efe368de.png)

    
#### 25. Explain the back propagation mechanism in dropout layers ?
    - During test time the weight are multiplied with dropout rate.
    
#### 26. Explain the loss function used in auto encoders assuming the network accepts images ?
    - Mean square error is used as loss function in autoencoder
    
#### 27. Numbers of tunable parameters in dropout layer ?
    - 0
    
#### 28. When F1 score will be zero? And why ?
    - If either precision or recall is 0.
    
#### 29. What is the need of dimensionality reduction.
    - Dimensionality reduction refers to techniques for reducing the number of input variables in training data. When
    dealing with high dimensional data, it is often useful to reduce the dimensionality by projecting the data to a 
    lower dimensional subspace which captures the “essence” of the data.
    
#### 30. What happens if we do not normalize our dataset before performing classification using KNN algorithm.
    - KNN performance usually requires preprocessing of data to make all variables similarly scaled and centered.
    - If we don’t normalize the data, all the features will be on different scales thus the model doesn’t Perform well.
    
#### 31. What is standard normal variate ?
    - Standard normal variate makes the data with 0 centering and with variance 1.
    
#### 32. What is the significance of covariance and correlation and in what cases can we not use correlation.

![Differences](https://user-images.githubusercontent.com/20341930/168098406-1b068f0c-fe5c-4479-8bcd-58d55e373f30.png)

    
#### 33. How do we calculate the distance of a point to a plane.
    
![Formula for Distance of point to Plane](https://user-images.githubusercontent.com/20341930/168099204-42c525f2-79ac-4bec-9110-825098db8859.png)
    
#### 34. When should we choose PCA over t-sne.
    
[Detailed Stats-Exchange Answer](https://stats.stackexchange.com/questions/238538/are-there-cases-where-pca-is-more-suitable-than-t-sne)
    
#### 35. How is my model performing if
```
    1. Train error and cross validation errors are high.
    2. Train error is low and cross validation error is high.
    3. Both train error and cross validation error are low.
```
    - Answer
      1. underfitting 
      2. overfitting 
      3. best fit
    
#### 36. How relevant / irrelevant is time based spitting of data in terms of weather forecasting ?
    - It is required to time-based splitting in weather forecasting as it changes over time.
    
#### 37. How is weighted knn algorithm better simple knn algorithm.
    - Weighted KNN takes the distance into account for K nearest neighbor by giving more weightage to points which are more near to query poin
    
#### 38. What is the key idea behind using a kdtree.
    - It is based on axis parallel lines and it is very useful in performing search queries.
    
#### 39. What is the relationship between specificity and false positive rate.
    - Solution:
    
![FPR VS Specificity](https://user-images.githubusercontent.com/20341930/168100158-17752904-e4ff-48f9-a48c-3d4cee473e5a.png)
    
#### 40. What is the relationship between sensitivity, recall, true positive rate and false negative rate?
    - Answer
    
#### 41. What is the alternative to using euclidean distance in Knn when working with high dimensional data ?
    - Solution:
    
![image](https://user-images.githubusercontent.com/20341930/168100299-603d43f4-937b-4e81-a828-43a0e5e18a6b.png)

    
#### 42. What are the challenges with time based splitting? How to check whether the train / test split will work or not for given distribution of data ?
    - Challenges:
        - Spliting it in same bin with different number of samples in each bins
        - Additional overhead of specific orders.
        - How long the window of time should be such that the distributions of test and train dataset is same.
    - How to split:
        - Split data in train and test set given a Date. Split train set in for example 10 consecutive time folds.
        - Then, in order not lo lose the time information, perform the following steps:
        - Train on fold 1 –>  Test on fold 2
        - Train on fold 1+2 –>  Test on fold 3
        - Train on fold 1+2+3 –>  Test on fold 4
        - Train on fold 1+2+3+4 –>  Test on fold 5
        - Train on fold 1+2+3+4+5 –>  Test on fold 6
        - Train on fold 1+2+3+4+5+6 –>  Test on fold 7
        - Train on fold 1+2+3+4+5+6+7 –>  Test on fold 8
        - Train on fold 1+2+3+4+5+6+7+8 –>  Test on fold 9
        - Train on fold 1+2+3+4+5+6+7+8+9 –>  Test on fold 10
        - Compute the average of the accuracies of the 9 test folds (number of folds  – 1)
    - Refer very good write up here: https://www.linkedin.com/pulse/time-based-splitting-determining-train-test-data-come-manraj-chalokia/?trk=public_profile_article_view
    
#### 43. How does outlies effect the performance of a model and name a few techniques to overcome those effects.
    - Random sample consensus (RANSAC) is an iterative method to estimate parameters of a mathematical model from a 
    set of observed data that contains outliers, when outliers are to be accorded no influence on the values of the estimates.
    - Anomaly detection Refer: LOF is one of the anomaly detection algorithm
    
#### 44. What is reachability distance?
    - Reachability distance. The k-distance is now used to calculate the reachability distance.
    - This distance measure is simply the maximum of the distance of two points and the k-distance of the second point. 
    - Basically, if point a is within the k neighbors of point b, the reach-dist(a,b) will be the k-distance of b.
    
![Reachability Distance](https://user-images.githubusercontent.com/20341930/168132037-722951e7-c234-466a-b00c-b2fdc6ba23fd.png)

    
#### 45. What is the local reachability density ?
    - The local reachability density is a measure of the density of k-nearest points around a point which is calculated
    by taking the inverse of the sum of all of the reachability distances of all the k-nearest neighboring points.
 
![Local Reachability distance](https://user-images.githubusercontent.com/20341930/168132135-84ce1720-ed89-47ea-8e91-cf21d033dba4.png)

  
#### 46. What is the need of feature selection ?
    - If the data is having very dimensional data the model would not perform well. So, to get the best model we
    do features selection which gets the useful features of data 
    And model is trained on selected model to yield better results.
    
#### 47. What is the need of encoding categorical or ordinal features?
    - Machine learning models require all input and output variables to be numeric. This means that if your data
    contains categorical data, you must encode it to numbers before you can fit and evaluate a model.
    
#### 48. What is the intuition behind bias-variance trade-off ?
    - Predictive models have a tradeoff between bias (how well the model fits the data) and 
    variance (how much the model changes based on changes in the inputs).

    - *Simpler models* are stable (low variance) but they don’t get close to the truth (high bias).

    - More *complex models* are more prone to being overfit (high variance) but they are expressive enough
    to get close to the truth (low bias).

    - The best model for a given problem usually lies somewhere in the middle.
    
#### 49. Can we use algorithm for real time classification of emails.
    - Yes
    
#### 50. What does it mean by precision of a model equal to zero is it possible to have precision equal to 0.
    - Yes, we can have 0 precision that means model doesn’t give any true positives.
    
---

#### Question_Template
#### X. Question
    - Answer
