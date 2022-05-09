#### 27. Which of these will have more numbers of tunable parameters?
```
1. (7,7,512) ⇒ flatern ⇒ Dense(512)
2. (7,7,512) ⇒ Conv (512,(7,7))
``` 
    - Answer
    
#### 28. What is overfitting and underfitting ? 
    - When your model gives near perfect results for all the training set but performs worst on test/unseen data then 
    we can say the model is overfitted to training set where it resembles training data set and was unable to identify 
    generalization for the data. Low bias & High variance is a signal to identify.
    
    - When your model does not provide accuracy neither on training nor on test dataset, it can be said that the given 
    model is simple. It doesn't learned anything and made very basic assumption for the data. Here it will be bias to one class.
    
    
#### 29. What do you do if a deep learning model is overfitting? 
    - Answer
    
#### 30. What is the batch Normalization layer ? 
    - Answer
    
#### 31. Write keras code to add a BN layer in an existing network ?
    - Answer
    
#### 32. Number of tunable parameters in the BN layer. 
    - Answer
    
#### 33. What is convolution operation? 
    - Answer
    
#### 34. Number of parameters in a convolution neural network given in architecture 
    - Answer
    
#### 35. What are the inputs required to calculate the average f1 score ? 
    - When we are working with binary classification we generally have only 1 F1 score.
    - However, when working with multiclass classification problem, we can't use F1-score directly as it is misleading in some sense due to weightage of clases.
    - So there are 3 F1 scores:
        1. Micro F1-Score => Accuracy
        2. Macro F1-Score => (Un-weighted mean of F1-Scores) Takes F1 score of all classes and average them. 
           Here individual F1-score will be calculate with One-VS-All.
        3. Avg F1-Score   => Takes weights/number of actual class labels into consideration while averaging F1-score
[When to use which?](https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f)
    
#### 36. What is the problem with macro average f1 score in 5 class classification problem?
    - Here Macro F1-Score => (Un-weighted mean of F1-Scores)
    - So if there is a dataset imbalance of any class it won't consider it. 
    - We calculate precision and recall for individual classes and do average of them
        P=P1+P2+P3+P4+P5/5
        R=R1+R2+R3+R4+R5/5
    - Now we calculate harmonic mean P and R to get macro-f1-score
    - Macro f1-score does not care for class imbalance data.
    
#### 37. How do you get probabilities for RF classifier outputs. 
    - In RF we will be having many base decision trees each predicting a class label.For probability of class in RF we calculate 
    total number of DT’s predicting the class Divided by total number of DT’s in RF.
    
#### 38. Is the Calibration classifier required to get probability values for logistic regression.? 
    - No, logistic regression gets the probabilty of the point being part of class. So applying calibration classifier is unnecessary.
    - In general, model which doesn't provide probabilty for classification, you should apply calibration class as it will impact 
    the 0.51 vs 0.99 probabiliry of the class and improve model in general for classification.
    
#### 39. How does kernel sum work in test time ? 
    - In kernel SVM we calculate the similarity of the points with the others in the train data with the help of kernel.
```
F(xq)=summation( [ αi * yi * kernel(xi ,xq) ] + b) for i=1 to n
```

#### 40. What kind of base learners are preferable in random forest classifiers ? 
    - In Random forest we prefer decision trees as the base learners because it can be trained very deep which can 
    have high variance easily.
    
#### 41. How does bootstraping works in RF classification. 
    - In random forest each base learner is trained on sample of data randomly from total data such that training many
    such models will see all the datapoints of train data.
    
#### 42. Difference between one vs rest and one vs one. 
    - The One-vs-Rest strategy splits a multi-class classification into one binary classification problem per class.
    - The One-vs-One strategy splits a multi-class classification into one binary classification problem per each pair of classes.
    
#### 43. Which one is better is one vs rest and one vs one. 
    - One vs rest is better because in One vs One we will take pairs of classes in each classification. So, in multiclass 
    there will be many such pairs which will make it difficult to calculate.
    
#### 44. What will happen if gamma increases in RBF kernel sum. 
    - As gamma in RBF kernel is increased more of the points are given some similarity as the range of distance for 
    similarity scores is increased. It is similar to K-NN with increased k values.
    
#### 45. Explain linear regression. 
    - Linear regression is an algorithm in which it tries to find a hyperplane which tries to fit to the training datapoints
    and has least sum of squared distances to the plane.
    
![Linear Regression equation](https://user-images.githubusercontent.com/20341930/166633126-ecff1db5-67f4-44bb-ab25-4cb839541298.png)

    
#### 46. What is difference between one hot encoding and a binary bow? 
    - In one hot encoding we take the number of times a category or word is repeated in the train set. When frequency matters.
    - In binary BOW we only take whether the category or word is present in train data (0 or 1). When presence matters.
    
#### 47. Kernal svm and linear svm ( SGD classifier with hinge loss). Whch has low latency and why?
    - Linear SVM is faster because in linear SVM no kernelization is involved. Thus linear tries to operate in lower dim only 
    where as kernel SVM tries to find higher dimensional features which takes time than linear SVM.
    
#### 48. Explain bayes theorem. 
    - The Bayes theorem describes the probability of an event based on the prior knowledge of the conditions that might be 
    related to the event. If we know the conditional probability , we can use the bayes rule to find out the reverse probabilities 

![Bayes Theorem](https://user-images.githubusercontent.com/20341930/167345317-119dbd7a-96d3-4ff6-943c-d52ee646362c.png)

    
#### 49. How to decrease the test time complexity of a logistic regression model. 
    - If we get a sparse Weight matrix. Creating the dictionary of Non-Zero feature weights and only calculate related 
    will eventually increase performance.
    
#### 50. What is the need for sigmoid function in logistic regression. 
    - Logistic regression works on the (Euclidean) distance of the points. So the distance is countably infinite numbers to keep 
    them in range.
    
![Optimization equation for LR](https://user-images.githubusercontent.com/20341930/167346370-44b2ecbf-7f62-4ba1-b1db-34804bb0e46b.png)

    
![2 plane for separation of 2 classes](https://user-images.githubusercontent.com/20341930/167346308-21c6c07d-f870-4395-9c9b-c253c52f7958.png)

    - Here based on distance, optimization equation gives this output.
    For plane-1 sum = 1+1+1+1+1+1+1+1–50 = -42
    For plane-2 sum = 1+2+3+4–1–2–3–4+1 = 1
    
    - However, Plane 1 is better than Plane 2. But due to outlier it is creating issue for plane selection And We can overcome this 
    by incorporating some function such that it has to squeeze the maximum distance points. one of such function is sigmoid. 
    
![Optimization Equation with Sigmoid](https://user-images.githubusercontent.com/20341930/167346749-a0f81418-ec3e-4f67-9130-2a038ddef0ea.png)


![Final Form of LR optimization Equation](https://user-images.githubusercontent.com/20341930/167346785-4d7602d4-9a7f-4d97-ab64-ed0c341a0a56.png)


---

#### Question_Template
#### X. Question
    - Answer
