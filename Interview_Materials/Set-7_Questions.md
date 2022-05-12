#### 1. What does trainable = true/false mean in embedding layer ?
    - Trainable=True means we train the weights by update equation.
    - Trainable=False freezes the weights without training.
    
#### 2. What happens when we set return sequence = true in LSTM ?
    - It tells whether to return last state of output or not.
    
#### 3. Why are RNN’S and CNN’S called weight shareable layers ?
    - A convolutional neural network learns certain features in images that are useful for classifying 
    the image. Sharing parameters gives the network the ability to look for a given feature everywhere 
    in the image, rather than in just a certain area. This is extremely useful when the object of interest
    could be anywhere in the image.
    - Relaxing the parameter sharing allows the network to look for a given feature only in a specific area. 
    For example, if your training data is of faces that are centered, you could end up with a network that 
    looks for eyes, nose, and mouth in the center of the image, a curve towards the top, and shoulders towards 
    the bottom.
    - It is uncommon to have training data where useful features will usually always be in the same area, so 
    this is not seen often.
    
#### 4. What happens during the fit and transform of following modules ?
```
    1. Standard scaler
    2. Count vectorizer
    3. PCA
```
    - Fit used to learn the vocabulary and creates a dictionary with key as word and Its words count in the documents as value.
    - Transform converts the given data to the representations that were learnt during training.
    - Fit-transform does the two steps on the same data simultaneously.
    
#### 5. Can we use t-sne for transforming test data ? if not why ?
    - The main reason that t-SNE is not used in classification models is that it does not learn a function from
    the original space to the new (lower) dimensional one. As such, when we would try to use our classifier on 
    new / unseen data we will not be able to map / pre- process these new data according to the previous t-SNE results.
    - t-SNE tries to find new distribution of data in lower space such that both the distributions are very similar. 
    - This is achieved by KL divergence.
    
#### 6. Find the sum of diagonals in the numpy array ?
    ``` 
    numpy.trace(array)
    ```
    
#### 7. Write the code to get the count of row for each category in the dataframes.
    ```
    df.groupby(columnname)[columnname].nunique()
    ```
    
#### 8. Difference between categorical cross entropy and binary cross entropy.
    - Categorical cross entropy is used for multi-class classification 
    - Whereas binary cross Entropy is used for two class classification.
    
#### 9. When you use w2v for test factorization, and we each sentence is having different words how can you forward data into models ?
    - Answer
    
#### 10. What is tf-idf weighted w2v ?
    - After obtaining w2v representation of a word we multiply it with a scalar value of tf-idf
    
#### 11. How to you use weighted distance in content based recommendation ?
    - Answer
    
#### 12. What is the time complexity of SVD decomposition ?
    - O(mn min(n, m))
    
#### 13. What is the difference between content based recommendation and collaborative recommendation ?
    - Content based RF users metadata and item metadata to get user and item vectors to compute similarities.
    - In collaborative filtering, we try to find user and items matrix using SVD
    
#### 14. Why do you think inertia actually works in choosing elbow point in clustering ?
    - Inertia: It is the sum of squared distances of samples to their closest cluster center.
    - If the use elbow method the inertia will abruptly change for the points out of cluster.
    
#### 15. What is gradient clipping ?
    - In training a neural network sometimes we may experience exploding gradients beacause of large weights that 
    exponentially increasing, to avoid this exploding gradient problem we use gradient clipping which limits the
    weight values with increasing.
    
#### 16. Which of these layers will be a better option as a last layer in multilabel classification ?
```
    1. Sigmoid
    2. Softmax  
```
    - Sigmoid because a point may have any number of classes.
    - In softmax we only get one class that have the max probability.
    
#### 17. Is there a relation or similarity between LSTM and RESNET ?
    - Yes, In LSTM we use forgot gate which enables the to control the amount of data to be forwarded. 
    - Similarly in RESNET we use residual connections which makes the data to flow from the residual connection 
    when weights are not useful.
    
#### 18. What are the values returned by np.histogram()
    - It takes a numpy nd array as input, it flattens the array and return histogram values which can be plotted
    to see the histogram of data.
    
#### 19. What is PDF, can we calculate PDF for discrete distribution ?
    - Yes
    
#### 20. Can the range of CDF be (0.5 - 1.5 ).
    - No max value of a cdf is 1. As it works on percentile of data.
    
#### 21. Number of parameters in the following network :
```
    1. Number of neurons = 4
    2. Problem = binary classification
    3. no: of FC = 2
    4. Neurons in 1st FC = 5
    5. Neurons in 2nd FC = 3
```
    - 54
    
#### 22. How do we interpret alpha in dual form of sum? What is the relation between C and Alpha?
    - alpha and C are inversely proportional to each other.
    
#### 23. How does back propagation work in case of LSTM?
    - Answer
    
#### 24. What is the difference between supervised and unsupervised models?
    - In supervised learning we have the class labels for the datapoints
    - In unsupervised learning we don’t have any class label data to train the model with.
    
#### 25. What is the derivative of this fraction 1/(1+e^sinx).
    - Answer is [(𝑒^(sin(𝑥)) (cos(𝑥) − 1))/((1 + 𝑒 ^ sin (𝑥) ))]
    - Try to work out it on pen and paper.
    
#### 26. What will be the output of 
    ```
    a = np.array([[1,2,3,10],[4,5,6,11],[7,8,9,12]])
    a[:,:-1]
    ```
    - array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])
    
#### 27. What is the output of this 
```
    a = np.array([[1,5,9],[2,6,10],[3,7,11],[4,8,12]])
    a[:-2,:]
```
    - array([[ 1,  5,  9],
             [ 2,  6, 10]])   
    
#### 28. What will be the output of
```
    a= dict()
    a[('a','b')] = 0
    a[(a,b)] = 1
    print(a)
```
    - b is not defined as the key tuple is string and update value failed for third statememt.
    -Correct one:
        ```
        a= dict()
        a[('a','b')] = 0
        a[('a','b')] = 1
        print(a)
        {('a', 'b'): 1}
        ```
    
#### 29. What will be the output of
```
    1. a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    2. np.mean( a,axis=1)   
```
    - array([2., 5., 8.])
    
    
#### 30. What will be the output of
```
    1.  a =[[3,4,5],[6,7,8],[9,10,11]]
    2.  b =[[1,2,3],[4,5,6],[7,8,9]]
    3.  np.stack( (a,b), axis= 0)
```
    - 
    ```
    array([[[ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11]],

           [[ 1,  2,  3],
            [ 4,  5,  6],
            [ 7,  8,  9]]])
    ```
    
#### 31. What is “local outlier factor”?
    - LOF is the anomaly detection technique used in K-NN
    
#### 32. How RANSAC works?
    - We take a sample of data from original data and train the model on sample data.
    - The probability of outlier in sampled data will be reduces. We will remove the outlier by Computing the loss with point. 
    - We remove the points which have high loss value from The original data 
    - And repeat the same process until the convergence.
    
#### 33. What are jaccard & Cosine Similarities
    - Answer
    
#### 34. What are assumptions of Pearson correlation ?
    - Linearity and absence of outliers.
    
#### 35. Differences between Pearson and Spearman correlation?
    - The fundamental difference between the two correlation coefficients is that
    - the Pearson coefficient works with a linear relationship between the two variables whereas
    - the Spearman Coefficient works with monotonic relationships as well.
    
#### 36. What is the train time complexity of DBSCAN?
    - The different complexities of the algorithm are(N= no of data points) as follows:

    - Best Case: If we use a spatial indexing system to store the dataset like kd-tree or r-tree such that 
    neighbourhood queries are executed in logarithmic time, then we get O(N log N) runtime complexity.

    - Worst Case: Without the use of index structure or on degenerated data the worst-case run time complexity remains O(N2).

    - Average Case: It is the same as the best/worst case depending on data and implementation of the algorithm
    
#### 37. Explain the procedure of “prediction in hierarchical clustering”
    - 
    
#### 38. Relation between knn and kernel SVM
    - Answer
    
#### 39. Proof of “convergence of kmeans”
    - Answer
    
#### 40. What is the optimal value of minpoints for the data (1000, 50)
    - Answer
    
#### 41. Why do you want to/ why did you choose data science as your career
    - Answer
    
#### 42. What is difference between AI, ML and DL?
    - Answer
    
#### 43. What is a Python Package, and Have you created your own Python Package?
    - Answer
    
#### 44. Can you write a program for inverted star program in python?
    - Answer
    
#### 45. Write a program to create a data frame and remove elements from it.
    - Answer
    
#### 46. Write code to find the 8th highest value in the Data Frame.
    - Answer
    
#### 47. What’s difference between an array and a list?
    - Answer
    
#### 48. Differentiate between Supervised, Unsupervised and Reinforcement learning with their algorithm example.
    - Answer
    
#### 49. How would you deal with feature of 4 categories and 20% null values?
    - Answer
    
#### 50. What is central tendency?
    - Answer
    
---

#### Question_Template
#### X. Question
    - Answer
