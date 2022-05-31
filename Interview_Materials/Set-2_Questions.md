#### 1. What is the optimization equation of GBDT ?
    - The objective function is usually defined in the following form:
![image](https://user-images.githubusercontent.com/20341930/164965638-374dad2a-b4fd-4a51-aea4-5f32dcd1f3d3.png)

    - Algorithm for GBDT: [We have to find the optimum value of gamma(γ) such that the value of loss reduces]
    
![image](https://user-images.githubusercontent.com/20341930/166418240-841c922c-20fb-4377-9b71-b48913c8fdfc.png)

[Want to checkout Example?](https://towardsdatascience.com/understanding-gradient-boosting-from-scratch-with-small-dataset-587592cc871f)

    - General question from given answer: What is pseudoresidual and why we use it?
    - It is a intermidiate error of the model. GBDT work on serial way current state is dependent on previous state of residual.
    - They help us minimise any loss function of our choice until and up till the loss function is differentiable
    
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
    - There are many forms of regularization: ℓ1, ℓ2, early stopping, dropout, etc. kNN is a nonparametric model,
    it doesn't have parameters to penalize, drop, or way to stop training earlier.
    
    - The main point of regularization is preventing overfitting. The way to prevent overfitting in kNN is 
    to increase k as it leads to averaging over many points instead of memorizing the training set as in k=1. 
    It also makes the result smoother, because of averaging, which is another common consequence of regularization. 
    A similar approach is used in decision trees where we restrict depth, minimal node size, or prune them, as means of regularization.
    
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
    
    - In naïve bayes if in the test data a category was not present in train data to avoid Problem of zero probability 
    we use Laplace smoothing. We add a small value (alpha) in Numerator and k*alpha in the denominator where k is the number 
    of values that a feature take.
    
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
    - alpha in Laplace smoothing will act as regularizer.
    
    - As alpha **increases** the likelihood probabilities will have uniform distribution i.e.  P(0)=0.5 and P(1)=0.5 thus 
    the prediction will be biased towards larger class.
    
    - As alpha **decreases** even with the small change in the data will affect the probability. Thus, it leads to overfitting.

![image](https://user-images.githubusercontent.com/20341930/164969720-8e83a253-3023-4182-add2-2f94b291a8a7.png)
    
#### 10. Can we solve dimensionality reduction with SGD?
    - Yes because dimensionality reduction can be posed as an optimization problem where we try to reduce the loss function.
```
(||(Xi - Xj)||^2 -Xij )^2
```
    - here xi and xj are points in higher dimesion and xij is the distance between the points in lower dimension thus we 
    prefer the neighborhood distance in the lower dimensions with minimal losses.
    
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
 
![Logistic Regression weight based optimization equation](https://user-images.githubusercontent.com/20341930/166428703-5195c485-2c11-4b11-865c-18738edfedfd.png)
    
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
    - Proportional sampling is the method of picking an element proportional to its weight, i.e., the higher the weight of the object, 
    the better are its chances of being selected.
    - Psudocode: 
    ```
    1. normailise all the values i.e the range will be (0,1)
    2. calculate the cummulative sum .
    3. Sample one ramdom value from unifrom distribution of (0,1)
    4. If the random value <= cummulative sum return the number
    ```

#### 16. What are hyperparameters in kernel svm ?
    - Be careful, There 2 form of SVM (Primal & Dual Form of SVM) 
    - In Primal, there is no kernelization where we can have lambda or C as in hyper parameter. C= 1/Lambda
    - In Dual Form of SVM kernel is itself a hyper parameter.
      Dual form of SVM it can have kernalized formulation of SVM optimization equation.
    - Based on the Kernel selected, there will be different appropriate hyperparameter. 
    - Example : In RBF kernel we have σ as hyperpameter.
[Please read this doc for checking out different kernels and their hyperparameter](https://philipppro.github.io/Hyperparameters_svm_/)
 
#### 17. What are hyperparameters in SGD with hinge loss ?
    - the regularization parameter (the alpha)
    - the number of iterations (n_iter)
 
#### 18. Is hinge loss differentiable? if not, how we will modify it so that you can apply SGD ?
    - Hinge loss is not differentiable at 1 as it is not continuous at 1. So, we use squared hinge loss in optimizations. 
    Or we can use smooth approximation for hinge loss as below.
    
![Modified version of hinge loss which is differentiable everywhere](https://user-images.githubusercontent.com/20341930/166430885-814f12e5-5eca-446d-9c01-293f18b9b810.png)

#### 19. What is RMSPROP?
    - RMSProp is Optimizer technique.
    - In general/SGD/MiniBatch SGD, we use single learning rate to converge to point. But it might converge to Local 
    Minima [Which is okay] or saddle point[Which is bad]. And at very constant slow pace
    - We are updating learning rate with each iteration to coverge faster using second order moment (variance).
    
![RMSProp](https://user-images.githubusercontent.com/20341930/171124605-021c6b26-2590-4975-bafb-baba6167cb60.png)
 
#### 20. What is ADAM ?
    - ADAM uses first order moment (mean) & second order moment (variance) both to update the learning rate in 
    runtime to converge faster.
 
![Adam](https://user-images.githubusercontent.com/20341930/171123450-7014bfa8-2833-4138-b657-fef8c3daf958.png)
 
#### 21. Difference between ADAM vs RMSPROP ?
    - Adam uses both First & Second Order Moment to update Learning rate.
    - While RMSProp uses Only second order moment.
    - Both of these are managed by Hyperparameter beta_1 (for mean) * beta_2 (for variance).
    - If beta_1 is 0 then ADAM == RMSProp.
    - For mathematical details please see above questions 19 & 20.
 

#### 22. What is the maximum and minimum values of gradient of the sigmoid function ?
    - Value of Sigmoid Function is between 0 to 1. So grediant => Slope of graph is shown as below.

![Sigmoid and its gredient](https://user-images.githubusercontent.com/20341930/166431265-615f377c-57ba-41f2-a437-136431d9d5ef.png)

 
#### 23. What is RELU? Is it differentiable ?
    - Relu is an activation function which can be used in NN.
    - Relu F(z) = Max(0, z) 

![Relu](https://user-images.githubusercontent.com/20341930/171125843-6408bcbf-92b6-41ff-925d-2aa2c1ae5928.png)

    - As you can see function in blue.
    - To be used in NN it needs to be Differenciable as the weight calculation works on derivative of activation function
    - But Relu is not differenciable at z=0.
    - So we are using Softplus function with similar effect to relu with Differenciable everywhere.
 
#### 24. What is precission and recall ?
    - Precision tells us that out the total positive points how many of them are predicted to be positive.
      Precision = TP/(TP + FP)
       
    - Recall tells us that out of total points that are predicted positive how many points are actually positive.
      Recall = TP/(TP + FN)
    
 
#### 25. What is F1 score ?
    - F1 score is a mathematical measure that combines Precision & Recall.
    
![image](https://user-images.githubusercontent.com/20341930/165241845-2c30e40f-be9e-4bd3-bd8e-7874ed482a21.png)

#### 26. Name a few weight initialization techniques ?
    - Random initialization
    - Taking values from specific distributions
    - Based on that perticular perceptron/neuron's input & output count [FanIn & FanOut]
    - Xavier 
    - HE initialization
    
![With Formulae](https://user-images.githubusercontent.com/20341930/171125055-d2c6cddf-fe03-4d88-9404-e6c8a07437e3.png)

 
---

#### Question_Template
#### X. Question
    - Answer
