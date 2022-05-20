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
    - 80% as more data the better.
    
#### 5. Explain bias – variance trade off. How does this affect the model?
    - What? => The bias–variance tradeoff is a central problem in supervised learning. Ideally, one wants to 
    choose a model that both accurately captures the regularities in its training data, but also generalizes 
    well to unseen data. 
    - Unfortunately, it is typically impossible to do both simultaneously. 
    - High-variance learning methods may be able to represent their training set well but are at risk of 
    overfitting to noisy or unrepresentative training data. 
    - In contrast, algorithms with high bias typically produce simpler models that may fail to capture important
    regularities (i.e. underfit) in the data.
    
    - Why? => If our model is too simple and has very few parameters then it may have high bias and low variance. 
    On the other hand if our model has large number of parameters then it’s going to have high variance and
    low bias. So we need to find the right/good balance without overfitting and underfitting the data.
    - This tradeoff in complexity is why there is a tradeoff between bias and variance. An algorithm can’t be more
    complex and less complex at the same time.
  
![Total error](https://user-images.githubusercontent.com/20341930/169347109-bb50cbfe-d579-4779-bd63-f9d33321ec51.png)

![Bias-Variance Tradeoff](https://user-images.githubusercontent.com/20341930/169346997-b93b39ee-2658-43ab-bef6-33840664beae.png)


    -An optimal balance of bias and variance would never overfit or underfit the model.
    
#### 6. What is multi collinearity? How to identify and remove it?
    - Multicollinearity occurs when independent variables in a regression model are correlated. This correlation 
    is a problem because independent variables should be independent. If the degree of correlation between variables
    is high enough, it can cause problems when you fit the model and interpret the results.
    
    - How to identify => VIF (Variance Inflation Factor) Value Range [1 -> Infinity)
        - 1-5 => Tolarable
        - >10 -> Needs work
        
    - To remove/decrease the affect =>
        - Structural Multicolinearity: Center you continuous variables
        - Data Multicolinearity: 
            - Remove highly correlated variables
            - Linearly combine correlated independent variables OR add them together
            - Use regularization
            
Refer: [Multicolinearity with regression](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/)
    
#### 7. Differentiate between Sensitivity and Specificity.
    - Sensitivity = [(TP/TP+FN)] x 100
    - Specificity = [(TN/TN+FP)] x 100
    - Example
   
![Sensitivity and Specificity](https://user-images.githubusercontent.com/20341930/169353378-23e2852f-f8c6-4841-964f-546665534cb6.png)

    
#### 8. What is difference between K-NN and K-Means clustering?
    - KNN => Supervised ML algo
    - KMeans => Unsupervised ML algo
    
#### 9. How to handle missing data? What imputation techniques can be used?
    - Drop
    - Imputation:
        - By Mean, Median, Mode, Any Aggregation
        - By using regression algorithms
    
#### 10. Explain how you would find and tackle an outlier in the dataset. Follow up: What about inlier?
    - Box plot [univariate analysis]
    - Percentiles and IQR
    - An inlier is a data observation that lies in the interior of a data set and is unusual or in error. 
    - Because inliers are difficult to distinguish from the other data values, they are sometimes difficult 
    to find and – if they are in error – to correct.
    
#### 11. How to determine if a coin is biased? Hint: Hypothesis testing
    - Null hypothesis,          Ho:P=0.5 (P=Q=0.5) [Not biased.]
    - Alternative Hypothysis:   H1:P>0.5
      where P is the prob of occuring head.
    - As there are success & failure, this is Binomial Distribution with n=900 and p=0.5 under the null hypothesis 
    (i.e. if the coin were unbiased then p=probability of heads(or tails) = 0.5) and Taking general Threshold p-value = 5%
    
    - We know z = (p-P)/sqrt(PQ/N)
    - Where p =490/900 =0.54

    - Now z=(0.54-0.5)/sqrt((0.5*0.5)/900)

    - z = 2.4
    - There is 2.4% chance that it will yield result that supports the Null Hypothysis. 
    - As 2.4% < Threshold selected 5%. Ho is rejected. 
    - Hence the coin is biased.....
    
#### 12. Is interpretability important for machine learning model? If so, ways to achieve interpretability for a machine learning models?
    - Yes, as it will explain why model classify/predicted pred value.
    - Otherwise ML model will black box and you can't decide and work on improving Model if not know
    how it came to given conclusion.
    - However, if interpretability is important within our algorithm — as it often is for high-risk environments
    then we must accept a tradeoff between accuracy and interpretability.
   
Refer: [Extensive Detailed Article with Math](https://towardsdatascience.com/guide-to-interpretable-machine-learning-d40e8a64b6cf)
    
#### 13. How would you design a data science pipeline?
    
![data Pipelien](https://user-images.githubusercontent.com/20341930/169357802-6cde0d11-5c0a-49ab-a3ad-41c499203db1.png)

    
#### 14. What does a statistical test do?
    - A statistical test provides a mechanism for making quantitative decisions about a process or processes. 
    The intent is to determine whether there is enough evidence to "reject" a conjecture or hypothesis about 
    the process. The conjecture is called the null hypothesis.
    
#### 15. Explain topic modelling in NLP and various methods in performing topic modelling.
    - Topic modelling refers to the task of identifying topics that best describes a set of documents. 
    These topics will only emerge during the topic modelling process (therefore called latent). And one 
    popular topic modelling technique is known as Latent Dirichlet Allocation (LDA).
    - Other methods and details: 

[Topic Modelling Methods](https://www.analyticssteps.com/blogs/what-topic-modelling-nlp)
    
#### 16. Describe back propagation in few words and its variants?
    - Backpropagation is an algorithm that back propagates the errors from output nodes to the input nodes. Therefore, 
    it is simply referred to as “backward propagation of errors”.  
    - Types of Backpropagation Network
    1. Static Backpropagation
    2. Recurrent Backpropagation
    [Back Propogation](https://www.elprocus.com/what-is-backpropagation-neural-network-types-and-its-applications/)

#### 17. Explain the architecture of CNN.
    Refer: [CNN Architecture](https://www.analyticsvidhya.com/blog/2020/10/what-is-the-convolutional-neural-network-architecture/)
    
![CNN](https://user-images.githubusercontent.com/20341930/169498996-7087cafd-9e72-4631-80a2-1bc705b80690.png)
 
    
#### 18. If we put a 3×3 filter over 6×6 image what will be the size of the output image?
    - Input: n X n and Filter Size is f X f, then the output size will be (n-f+1) X (n-f+1)
    => 4x4
    
#### 19. What will you do to reduce overfitting in deep learning models?
    - We can solve the problem of overfitting by:
    1. Increasing the training data by data augmentation
    2. Feature selection by choosing the best features and remove the useless/unnecessary features
    3. Early stopping the training of deep learning models where the number of epochs is set high
    4. Dropout techniques by randomly selecting nodes and removing them from training
    5. Reducing the complexity of the model
    
#### 20. How would you check if the model is suffering from multi-Collinearity?
    - VIF (Variance Inflation Factor) Value Range [1 -> Infinity)
        - 1-5 => Tolarable
        - >10 -> Needs work
    
#### 21. Why is CNN architecture suitable for image classification and not an RNN?
    - While RNNs are suitable for handling temporal or sequential data, CNNs are suitable for handling 
    spatial data (images). Though both models work a bit similarly by introducing sparsity and reusing 
    the same neurons and weights over time (in case of RNN) or over different parts of the image 
    (in case of CNN).
    
#### 22. What are the approaches for solving class imbalance problem?
    - Upsampling
    - Downsampling
    
#### 23. Tell us about transfer learning? What are the steps you would take to perform transfer learning?
    - Transfer learning is a machine learning method where we reuse a pre-trained model as the 
    starting point for a model on a new task.
    - To put it simply—a model trained on one task is repurposed on a second, related task as an optimization
    that allows rapid progress when modeling the second task.
    - By applying transfer learning to a new task, one can achieve significantly higher performance than 
    training with only a small amount of data.
    
    - How? => For Classical & Deep Learning both: https://www.v7labs.com/blog/transfer-learning-guide
    
    
#### 24. Explain concepts of epoch, batch, and iteration in deep learning.
    - The batch size is a hyperparameter of gradient descent that controls the number of training samples 
    to work through before the model’s internal parameters are updated.
    - The number of epochs is a hyperparameter of gradient descent that controls the number of complete passes 
    through the training dataset.
    - An iteration in deep learning, is when all of the batches are passed through the model.
    
    
#### 25. When sampling, what types of biases can be inflected? How to control the biases?
    - [Answer](https://www.scribbr.com/methodology/sampling-bias/)
    
#### 26. What are some of the types of activation functions and specifically when to use them?
    - Popular types of activation functions and when to use them
    1. Binary Step
    2. Linear
    3. Sigmoid
    4. Tanh
    5. ReLU
    6. Leaky ReLU
    7. Parameterised ReLU
    8. Exponential Linear Unit
    9. Swish
    10. Softmax
    
    -Which one to use?
    1. Sigmoid functions and their combinations generally work better in the case of classifiers
    2. Sigmoids and tanh functions are sometimes avoided due to the vanishing gradient problem
    3. ReLU function is a general activation function and is used in most cases these days
    4. If we encounter a case of dead neurons in our networks the leaky ReLU function is the best choice
            Always keep in mind that ReLU function should only be used in the hidden layers
    5. As a rule of thumb, you can begin with using ReLU function and then move over to other activation 
    functions in case ReLU doesn’t provide with optimum results
    
#### 27. Tell us the conditions that should be satisfied for a time series to be stationary?
    - Stationary implies that samples of identical size have identical distribution. This form is very restrictive, 
    and we rarely observe it, so for doing TSA, the term “stationarity” is used to describe covariance stationarity.
    
    - To some time series to be classified as stationary (covariance stationarity), it must satisfy 3 conditions:
    1. Constant mean
    2. Constant variance
    3. Constant covariance between periods of identical distance
    
    
#### 28. What is the difference between Batch and Stochastic Gradient Descent?
![Comparision BGD VS SGD](https://user-images.githubusercontent.com/20341930/169361405-9c6c1ca0-1bd8-4f93-a16c-40b8dfee4791.png)
    
#### 29. What happens when neural nets are too small? Tell us, What happens when they are large enough?
    - When we initialize weights too small(<1)? Their gradient tends to get smaller as we move backward through 
    the hidden layers, which means that neurons in the earlier layers learn much more slowly than neurons in 
    later layers. This causes minor weight updates.
    - Large NN? 
    
#### 30. Why do we need pooling layer in CNN? Common pooling methods?
    - Pooling layers are used to reduce the dimensions of the feature maps. Thus, it reduces the number of 
    parameters to learn and the amount of computation performed in the network. The pooling layer summarises 
    the features present in a region of the feature map generated by a convolution layer.
    
    - Two common functions used in the pooling operation are:
       1. Average Pooling: Calculate the average value for each patch on the feature map.
       2. Maximum Pooling (or Max Pooling): Calculate the maximum value for each patch of the feature map.
    
#### 31. Are ensemble models better than individual models? 
    -  There is no absolute guarantee a ensemble model performs better than an individual model every time, but 
    if you build many of those, and your individual classifier is weak. Your overall performance should be 
    better than an individual model. 
    
#### 32. How is random forest different from Gradient boosting algorithm, given both are tree-based algorithm?
    - Like random forests, gradient boosting is a set of decision trees. The two main differences are: How trees 
    are built: random forests builds each tree independently while gradient boosting builds one tree at a time.
    
#### 33. Is XOR data linearly separable?
    - Not linearly separable
    
#### 34. How do we classify XOR data using logistic regression?
    Refer: [Famous XOR problem](https://datahacker.rs/006-solving-the-xor-problem-using-neural-networks-with-pytorch/#4.-The-XOR-Problem:-Formal-Solution)
    
#### 35. LSTM solves the vanishing gradient problem that RNN primarily have. How?
    Refer: [Medium Article](https://medium.datadriveninvestor.com/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577)
    
#### 36. GRU is faster compared to LSTM. Why?
    - In terms of model training speed, GRU is 29.29% faster than LSTM for processing the same dataset; and 
    in terms of performance, GRU performance will surpass LSTM in the scenario of long text and small dataset,
    and inferior to LSTM in other scenarios.
    
    - GRU has advantages over long short term memory (LSTM). GRU uses less memory and is faster than LSTM, 
    however, LSTM is more accurate when using datasets with longer sequences.
    
#### 37. I have 2 guns with 6 holes in each, and I load a single bullet In each gun, what is the probability that if I fire the guns simultaneously,at least 1 gun will fire (at least means one or more than one)?
    - 11/18 [Probability of 1 fires and another not * 2 + Both Fires]
    - [(1/6 * 5/6) + (5/6 * 1/6) + (1/6 * 1/6) ]
    
#### 38. There are 2 groups g1 and g2, g1 will ask g2 members to give them 1 member so that they both will be equal in number, g2 will ask g1 members to give them 1 member so that they will be double in number of g1, how many members are there in each group?
    - G1 has x members & G2 has y members currently
    - Equ 1: x+1 = y-1
    - Equ 2: y+1 = 2(x-1)
    
#### 39. Tell the Order of execution of an SQL query.
    - Six Operations to Order: SELECT, FROM, WHERE, GROUP BY, HAVING, and ORDER BY.
    
#### 40. SQL Questions – Group by Top 2 Salaries for Employees – use Row num and Partition.
    - SELECT temp.salary FROM
                (
                    SELECT DISTINCT(salary) 
                    ROW_NUMBER() OVER(PARTITION BY ORDER BY salary DESC) as row_num
                    FROM salary 
                 ) AS temp
      WHERE temp.row_num<3
    
#### 41. Differentiate between inner join and cross join.
    - Inner join returns (A n B).
    - Cross join return (A u B) 
    
    ![Joins](https://user-images.githubusercontent.com/20341930/169513915-b7a82e80-e72a-48ac-a02b-60b8c0c1ff44.png)

    
#### 42. What is group-by?
    - GROUP BY is used to merge the other column results based on specific columns.
    
#### 43. Complex sql query– 2 table are there, Table1 (cust_id,Name) and Table2(cust_id,Transaction_amt). Write a query to return the name of customers with 8th highest lifetime purchase. Achieve the same using python.
    - Table1 (cust_id,Name) and Table2(cust_id,Transaction_amt)
    - SELECT IF(t1.name, t1.name, "Not Found") AS "8th_Customer"
      FROM Table1 as t1
      INNER JOIN
        ( 
          SELECT cust_id, sum(Transaction_amt) as purchases 
          FROM Table2
          GROUP BY cust_id
          ORDER BY purchases
          LIMIT 1 OFFSET 7
        ) AS t2
       ON t1.cust_id = t2.cust_id
    
#### 44. How is SVM(RBF-SVM) related to KNN?
    - 1. In Kernel function RBF (Radial Basis Function) the hyper parameter Alpha behaves exactly same as K in K-NN. 
    - 2. As the distance between 2 points increases it will similarity decreses.
    - 3. Comparible but in KNN, you need to store all dataset and in SVM you need just Support Vectors only.
    
#### 45. How is SVM related to Logistic Regression? 
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
   
---

#### Question_Template
#### X. Question
    - Answer
