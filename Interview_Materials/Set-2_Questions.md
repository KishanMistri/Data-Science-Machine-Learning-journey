#### 1. What is the optimization equation of GBDT ?
    - The objective function is usually defined in the following form:
![image](https://user-images.githubusercontent.com/20341930/164965638-374dad2a-b4fd-4a51-aea4-5f32dcd1f3d3.png)
    
    - Among them, L is a loss function, which is used to measure the quality of model fitting training data; Omega is called 
    a regular term, which is used to measure the complexity of the learned model.
    
    - The GBDT algorithm can be regarded as an addition model consisting of K trees.
![image](https://user-images.githubusercontent.com/20341930/164965693-c7c026d9-9f2b-42ad-9e49-9a16a3aa043f.png)
![image](https://user-images.githubusercontent.com/20341930/164965735-b9c102b0-4cc3-41aa-8517-048f5afaf2b2.png)

![image](https://user-images.githubusercontent.com/20341930/164965777-f1143c00-41ab-4d04-9619-fa3bd6f7bd08.png)

    - For example, assuming that the loss function is square loss, the objective function is:
![image](https://user-images.githubusercontent.com/20341930/164965782-d37641ca-aea0-47f1-86dd-5bc78ddba8a9.png)

    - Among them, It is residual. Therefore, when using the square loss function, each step of the GBDT algorithm only needs to fit 
    the residual of the previous model when generating the decision tree.   
![image](https://user-images.githubusercontent.com/20341930/164965813-a9411c08-00ba-45ce-8b54-cc74e0fc7f3e.png)

    
#### 2. Write the formulation of hinge loss ?
    - Hinge loss is written as 
      ```
      max(0, 1-Zi) ⇒ Type of Slack Variable
      Where Zi = Yi * (W^TXi + b) Yi => {1,-1}
      ```
    - For Correctly Classify point:   Zi > 1   → 1- Zi <0 so Hinge Loss = 0
    - For Incorrectly Classify point: Zi < 1   → 1- Zi >0 so Hinge Loss = 1-Zi which will penalize
    
#### 3. What is the train time complexity of KNN ?
    - Train:
        Time Complexity: O(nd)
        Space Complexity: O(nd) 
        
#### 4. What is the Test time complexity of KNN in brute force ?
    - Training time complexity: O(1)
    - Training space complexity: O(1)
    - Prediction time complexity: O(k * n * d)
    - Prediction space complexity: O(1)
    - Training phase technically does not exist, since all computation is done during prediction, so we have O(1) for both time and space.
    
#### 5. What is the test time complexity of KNN if we use kd-tree ?
    - Training time complexity: O(d * n * log(n))
    - Training space complexity: O(d * n)
    - Prediction time complexity: O(k * log(n))
    - Prediction space complexity: O(1)
    - During the training phase, we have to construct the k-d tree. 
    
#### 6. How will you regularize the KNN model ?
    - Answer
    
#### 7. Which of these model are preferable when we have low complexity power ? [While predicting - In general cases]
      1. SVM      2. KNN      3. Linear Regressions       4. XGBoost
    
    -> 1. Linear Regression      2. SVM       3. XGBoost        4. KNN
    - n         the number of training sample, 
    - p         the number of features, 
    - n_trees   the number of trees (for methods based on various trees), 
    - n_sv      the number of support vectors and 
    - nl_i      the number of neurons at layer 
    - i         in a neural network, we have the following approximations.
    
![image](https://user-images.githubusercontent.com/20341930/164970201-54512a43-20c5-4966-917c-d99acad63a09.png)


#### 8. What is Laplace smoothing ?
    - Laplace Smoothing / Additive Smoothing: [Not Laplacian Smoothing used in Image processing]
    - Q -> What will you do if the test word/feature is not present in train data?
    - Because without its probability, Total multiplication with this feature is 0 or 1 with respect to the class.

![image](https://user-images.githubusercontent.com/20341930/164969625-b01e6fdb-5f00-46e2-ac90-ef51ec98ba29.png)

![image](https://user-images.githubusercontent.com/20341930/164969522-13580f48-895e-4c04-80b2-9d9726295837.png)
        
        - Wi    is new/existing word
        - N     total number of points in set
        - alpha randomly selected constant
        - d     Set of Values for given Xi features
    - It will be applied to all points. It will work on points which are not present and give them avg probability (1/num of Classes)
    - It will result in ***Uniform distribution*** for your data points.
    - Technique used to [smooth](https://en.wikipedia.org/wiki/Smoothing) [categorical data](https://en.wikipedia.org/wiki/Categorical_data)
    
   
#### 9. How will you regularise your naive bayes model ?
    - Handle Missing Values if there are any.
    - Use Boosting if you want high performance as well.

![image](https://user-images.githubusercontent.com/20341930/164969720-8e83a253-3023-4182-add2-2f94b291a8a7.png)

    
#### 10. Can we solve dimensionality reduction with SGD?
    - Yes
    - Explain LDA (Minimize distance of same class & Maximize distance with other class) & PCA basic (Minimize distances to 
    other points, so no class level segregation in PCA)
    - https://stats.stackexchange.com/questions/427339/how-can-one-implement-pca-using-gradient-descent
    
#### 11. Which of these will be doing more computations GD or SGD ?
    - In both gradient descent (GD) and stochastic gradient descent (SGD), you update a set of parameters in an iterative
    manner to minimize an error function.
    
    - While in GD, you have to run through ALL the samples in your training set to do a single update for a parameter in 
    a particular iteration, in SGD, on the other hand, you use ONLY ONE or SUBSET of training sample from your training set 
    to do the update for a parameter in a particular iteration. If you use SUBSET, it is called Minibatch Stochastic gradient Descent.
    
    - Thus, if the number of training samples are large, in fact very large, then using gradient descent may take too long because 
    in every iteration when you are updating the values of the parameters, you are running through the complete training set. On the 
    other hand, using SGD will be faster because you use only one training sample and it starts improving itself 
    right away from the first sample.
    
    - SGD often converges much faster compared to GD but the error function is not as well minimized as in the case of GD. Often in 
    most cases, the close approximation that you get in SGD for the parameter values are enough 
    because they reach the optimal values and keep oscillating there.
    
    - If you need an example of this with a practical case, check Andrew NG's notes here where he clearly shows you the steps involved 
    in both the cases. [cs229-notes](https://web.archive.org/web/20180618211933/http://cs229.stanford.edu/notes/cs229-notes1.pdf)
    
#### 12. If A is a matrix of size (3,3) and B is a matrix of size (3,4) how many numbers of multiplications that can happen in the operations A*B ?
    - [A]i,j [B] j,k => [AB]i,k
    - Multiplication => i x j x k
    - 3 x 3 x 4 => 36
    
#### 13. What is the optimization equation of Logistic Regression ?
    - Answer
    
#### 14. How will you calculate the P(x|y=0) in case of gaussian naive baiyes ?
    - Here I have Probability of X ,given label/Class of Y = 0
    - Get the set of data where label Y == 0, Checking how much of them has value X for feature. That will be the Answer
    - Example:
      
![image](https://user-images.githubusercontent.com/20341930/164971163-e3a1e941-5f03-47c4-b52a-2f76d0eae53a.png)
 
    - Here Play tennis Yes=1 & No=0
    - P(Y=1) = 9/14 P(Y=0) = 5/14
    - Let's say feature X = Wind (1= Strong & 0 = Week)
    - P(X=1 | Y=0) = Probability of Strong Wind(x), given that we don't want play tennis outside => 3/5
    
#### 15. Write the code for proportional sampling.
    - Proportional sampling is the method of picking an element proportional to its weight, i.e., the higher the weight of the object, the better are its chances of being selected.
    - Code: 
    ```
    ```

#### 16. What are hyperparameters in kernel svm ?
    - Answer
 
#### 17. What are hyperparameters in SGD with hinge loss ?
    - Answer
 
#### 18. Is hinge loss differentiable if not how we will modify it so that you apply SGD ?
    - Answer
 
#### 19. Difference between ADAM vs RMSPROP ?
    - Answer
 
#### 20. What is RMSPROP?
    - Answer
 
#### 21. What is ADAM ?
    - Answer
 
#### 22. What is the maximum and minimum values of gradient of the sigmoid function ?
    - Answer
 
#### 23. What is RELU? Is it differentiable ?
    - Answer
 
#### 24. What is F1 score ?
    - Answer
 
#### 25. What is precission and recall ?
    - Answer
 
#### 26. Name a few weight initialization techniques ?
    - Answer
 

   
###### References/Sources:
1. [AppliedRoot](https://appliedroots.com/)

---

#### Question_Template
#### X. Question
    - Answer
