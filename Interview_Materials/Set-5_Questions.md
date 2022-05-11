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
    - Answer
    
#### 7. How do you compute the feature importance in DT ?
    - Answer
    
#### 8. How do you compute the feature importance in SVM ?
    - Answer
    
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
    - Regularization term = $(\lambda_1 * |W_i|_1) + (\lambda_2 * |W_i|_2)$
    
#### 12. What are the assumption of NB ?
    - That all the features are conditionally independent. 
    Which is not seen in real world, every feature has some kind of dependency (minimum/very low).
    
#### 13. What are the assumptions of KNN ?
    - KNN uses neighborhood so basic assumption is that the point/data point in the close proximity is the basis of same class classification
    
#### 14. What are the assumptions of linear regression ?
    - Answer
    
#### 15. Write the optimization equation of linear regression ?
    - Answer
    
#### 16. What is time complexity of building KD tree ?
    - Answer
    
#### 17. What is the time complexity to check if a number is prime or not ?
    - Answer
    
#### 18. Angle between two vectors (2,3,4) (5,7,8).
    - Answer
    
#### 19. Angle between the weigh vector of 2x+3y+5=0 and the vector(7,8).
    - Answer
    
#### 20. Distance between (7,9) and the line 7x+4y-120=0.
    - Answer
    
#### 21. Distance between the lines 4x+5y+15=0, 4x+5y-17=0.
    - Answer
    
#### 22. 122. Which of this hyperplane will classify these two class points
```
    1. P: (2,3), (-3,4) N: (-5,7), (-5,-9)
    2. 4x+5y+7=0, -3y+3x+9=0
```
    - Answer
    
#### 23. 123. Which of the vector pairs perpendicular to each other
```
    1. (3,4,5) (-3,-4,5)
    2. (7,4,6) (-4,-7,-12)
```
    - Answer
    
#### 24. How dropout works ?
    - Answer
    
#### 25. Explain the back propagation mechanism in dropout layers ?
    - Answer
    
#### 26. Explain the loss function used in auto encoders assuming the network accepts images ?
    - Answer
    
#### 27. Numbers of tunable parameters in dropout layer ?
    - Answer
    
#### 28. When F1 score will be zero? And why ?
    - Answer
    
#### 29. What is the need of dimensionality reduction.
    - Answer
    
#### 30. What happens if we do not normalize our dataset before performing classification using KNN algorithm.
    - Answer
    
#### 31. What is standard normal variate ?
    - Answer
    
#### 32. What is the significance of covariance and correlation and in what cases can we not use correlation.
    - Answer
    
#### 33. How do we calculate the distance of a point to a plane.
    - Answer
    
#### 34. When should we choose PCA over t-sne.
    - Answer
    
#### 35. 135. How is my model performing if
```
    1. Train error and cross validation errors are high.
    2. Train error is low and cross validation error is high.
    3. Both train error and cross validation error are low.
```
    - Answer
    
#### 36. How relevant / irrelevant is time based spitting of data in terms of weather forecasting ?
    - Answer
    
#### 37. How is weighted knn algorithm better simple knn algorithm.
    - Answer
    
#### 38. What is the key idea behind using a kdtree.
    - Answer
    
#### 39. What is the relationship between specificity and false positive rate.
    - Answer
    
#### 40. What is the relationship between sensitivity,recall,true positive rate and false negative rate?
    - Answer
    
#### 41. What is the alternative to using euclidean distance in Knn when working with high dimensional data ?
    - Answer
    
#### 42. What are the challenges with time based splitting? How to check whether the train / test split will work or not for given distribution of data ?
    - Answer
    
#### 43. How does outlies effect the performance of a model and name a few techniques to overcome those effects.
    - Answer
    
#### 44. What is reachability distance?
    - Answer
    
#### 45. What is the local reachability density ?
    - Answer
    
#### 46. What is the need of feature selection ?
    - Answer
    
#### 47. What is the need of encoding categorical or ordinal features?
    - Answer
    
#### 48. What is the intuition behind bias-variance trade-off ?
    - Predictive models have a tradeoff between bias (how well the model fits the data) and 
    variance (how much the model changes based on changes in the inputs).

    - *Simpler models* are stable (low variance) but they don’t get close to the truth (high bias).

    - More *complex models* are more prone to being overfit (high variance) but they are expressive enough
    to get close to the truth (low bias).

    - The best model for a given problem usually lies somewhere in the middle.
    
#### 49. Can we use algorithm for real time classification of emails.
    - Answer
    
#### 50. What does it mean by precision of a model equal to zero is it possible to have precision equal to 0.
    - Answer
    
---

#### Question_Template
#### X. Question
    - Answer
