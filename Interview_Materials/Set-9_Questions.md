#### 1. How to deal with imbalance data in classification modelling?
    - Downsampling [Random sampling]
    - Upsampling [SMOTE]
    
#### 2. What is Gradient Descent? What is Learning Rate and why we need to reduce or increase? Tell us why Global minimum is reached and why it doesn’t improve when increasing the LR after that point?
    - Gradient descent is an optimization algorithm which is commonly-used to train machine learning models and neural networks.
    - Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with 
    respect the loss gradient. The lower the value, the slower we travel along the downward slope. While this might
    be a good idea (using a low learning rate) in terms of making sure that we do not miss any local minima, it 
    could also mean that we’ll be taking a long time to converge — especially if we get stuck on a plateau region.
    
    ```
    new_weight = existing_weight — learning_rate * gradient
    ```
    - If learning rate is high it will have deflective effect on finding minima/maxima.
    
    ![Learning Rate effect in GBD](https://user-images.githubusercontent.com/20341930/169075488-6329350d-7a87-4cc0-97e1-9d716f50d1c1.png)

    
    
#### 3. What is Log-Loss and ROC-AUC?
    - Checkout what, why we need LOG-LOSS where ROC-AUC fails/limits.
    - [Answer](https://www.datamachines.io/blog/auc-vs-log-loss)
    
#### 4. Two Logistic Regression Models – Which one will you choose – One is trained on 70% and other on 80% data. Accuracy is almost same?
    - Answer
    
#### 5. Explain bias – variance trade off. How does this affect the model?
    - Answer
    
#### 6. What is multi collinearity? How to identify and remove it?
    - Answer
    
#### 7. Differentiate between Sensitivity and Specificity.
    - Answer
    
#### 8. What is difference between K-NN and K-Means clustering?
    - Answer
    
#### 9. How to handle missing data? What imputation techniques can be used?
    - Answer
    
#### 10. Explain how you would find and tackle an outlier in the dataset. Follow up: What about inlier?
    - Answer
    
#### 11. How to determine if a coin is biased? Hint: Hypothesis testing
    - Answer
    
#### 12. Is interpretability important for machine learning model? If so, ways to achieve interpretability for a machine learning models?
    - Answer
    
#### 13. How would you design a data science pipeline?
    - Answer
    
#### 14. What does a statistical test do?
    - Answer
    
#### 15. Explain topic modelling in NLP and various methods in performing topic modelling.
    - Answer
    
#### 16. Describe back propagation in few words and its variants?
    - Answer
    
#### 17. Explain the architecture of CNN.
    - Answer
    
#### 18. If we put a 3×3 filter over 6×6 image what will be the size of the output image?
    - Answer
    
#### 19. What will you do to reduce overfitting in deep learning models?
    - Answer
    
#### 20. How would you check if the model is suffering from multi-Collinearity?
    - Answer
    
#### 21. Why is CNN architecture suitable for image classification and not an RNN?
    - Answer
    
#### 22. What are the approaches for solving class imbalance problem?
    - Answer
    
#### 23. Tell us about transfer learning? What are the steps you would take to perform transfer learning?
    - Answer
    
#### 24. Explain concepts of epoch, batch, and iteration in deep learning.
    - Answer
    
#### 25. When sampling, what types of biases can be inflected? How to control the biases?
    - Answer
    
#### 26. What are some of the types of activation functions and specifically when to use them?
    - Answer
    
#### 27. Tell us the conditions that should be satisfied for a time series to be stationary?
    - Answer
    
#### 28. What is the difference between Batch and Stochastic Gradient Descent?
    - Answer
    
#### 29. What happens when neural nets are too small? Tell us, What happens when they are large enough?
    - Answer
    
#### 30. Why do we need pooling layer in CNN? Common pooling methods?
    - Answer
    
#### 31. Are ensemble models better than individual models? Why/why – not?
    - Answer
    
#### 32. How is random forest different from Gradient boosting algorithm, given both are tree-based algorithm?
    - Answer
    
#### 33. Describe steps involved in creating a neural network?
    - Answer
    
#### 34. In brief, how would you perform the task of sentiment analysis?
    - Answer
    
#### 35. Is XOR data linearly separable?
    - Answer
    
#### 36. How do we classify XOR data using logistic regression?
    - Answer
    
#### 37. LSTM solves the vanishing gradient problem that RNN primarily have. How?
    - Answer
    
#### 38. GRU is faster compared to LSTM. Why?
    - Answer
    
#### 39. Use Case – Consider you are working for pen manufacturing company. How would you help sales team with leads using Data analysis?
    - Answer
    
#### 40. I have 2 guns with 6 holes in each, and I load a single bullet In each gun, what is the probability that if I fire the guns simultaneously,at least 1 gun will fire (at least means one or more than one)?
    - Answer
    
#### 41. There are 2 groups g1 and g2, g1 will ask g2 members to give them 1 member so that they both will be equal in number, g2 will ask g1 members to give them 1 member so that they will be double in number of g1, how many members are there in each group?
    - Answer
    
#### 42. Tell the Order of execution of an SQL query.
    - Answer
    
#### 43. SQL Questions – Group by Top 2 Salaries for Employees – use Row num and Partition.
    - Answer
    
#### 44. Differentiate between inner join and cross join.
    - Answer
    
#### 45. What is group-by?
    - Answer
    
#### 46. Complex sql query– 2 table are there, Table1 (cust_id,Name) and Table2(cust_id,Transaction_amt). Write a query to return the name of customers with 8th highest lifetime purchase. Achieve the same using python.
    - Answer
    
#### 47. How is SVM(RBF-SVM) related to KNN?
    - 1. In Kernel function RBF (Radial Basis Function) the hyper parameter Alpha behaves exactly same as K in K-NN. 
    - 2. As the distance between 2 points increases it will similarity decreses.
    - 3. Comparible but in KNN, you need to store all dataset and in SVM you need just Support Vectors only.
    
#### 48. How is SVM related to Logistic Regression? 
    1. Logistic Regression SVM where no Kernelization used and just regular Xi & Xj thats where 
    it is behaving same as LR with different optimization function
    
    2. Both Linear SVM and Logistic Regression belong to the class of Linear Classifiers.
    
    3. Logistic Regression minimizes the 'Log' loss whereas SVM minimizes the 'Hinge' loss.
    
    4. Logistic Regression deals with finding out a linear surface that could separate the points 
    belonging to different classes, whereas Linear SVM deals with not only finding out a linear surface 
    that could separate the points belonging to different classes, but also maximizes the margin.
    
    5. Logistic Regression uses only Gradient Descent (or any of the other variants of Gradient Descent 
    like SGD, Batch-SGD) algorithm for optimization, whereas SVM supports Gradient Descent (or any of the other 
    variants of Gradient Descent like SGD, Batch-SGD) and also LibSVM and LibLinear libraries.
    
    6. Logistic Regression yields probabilistic estimates directly, whereas SVM doesn't yield probabilistic estimates 
    directly(when LibSVM/LibLinear are used). SVM can yield probabilistic estimates only if Gradient Descent 
    (or any of the other variants of Gradient Descent like SGD, Batch-SGD) is used, with calibration applied after 
    fitting the model on the data.
    
    7. Logistic Regression has an extension for multi-class (called softmax classifier, which is going to be discussed 
    in the Deep Learning), whereas SVM cannot be extended to a multi-class.
    
    8. SVM has a variant called Kernel SVM that can be applied directly on Non linear data (Feature transformations are 
    taken care by Kernel SVM internally), whereas Logistic Regression also has a Kernel version which was not popular, and 
    also not used in the industry. Kernel Logistic Regression wasn't implemented in scikit-learn as well.
    
    9. When we compare both the optimization problems of the Logistic Regression and the primary form of Linear SVM, 
    we can say ('C' is a hyperparameter that has to be tuned)
    
    10. In Logistic Regression,
        a) If 'C' is large --> Model Underfits    
        b) If 'C' is small --> Model Overfits
        
        In Linear SVM,
        a) If 'C' is large --> Model Overfits
        b) If 'C' is small --> Model Underfits
    
#### 49. 
    - Answer
    
#### 50. 
    - Answer
    
---

#### Question_Template
#### X. Question
    - Answer
