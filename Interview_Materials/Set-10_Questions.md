#### 1. List the most popular distribution curves along with scenarios where you will use them in an algorithm.
    - The most popular distribution curves are as follows- Bernoulli Distribution, Uniform Distribution, 
    Binomial Distribution, Normal Distribution, Poisson Distribution, and Exponential Distribution.
    - Each of these distribution curves is used in various scenarios.

    - Bernoulli Distribution can be used to check if 
      - a team will win a championship or not, 
      - a newborn child is either male or female, 
      - you either pass an exam or not, etc.

    - Uniform distribution is a probability distribution that has a constant probability. 
      - Rolling a single dice is one example because it has a fixed number of outcomes.

    - Binomial distribution is a probability with only two possible outcomes, the prefix ‘bi’ means two or twice. 
      - An example of this would be a coin toss. The outcome will either be heads or tails.

    - Normal distribution describes how the values of a variable are distributed. It is typically a symmetric 
    distribution where most of the observations cluster around the central peak. The values further away from 
    the mean taper off equally
    in both directions. 
      - An example would be the height of students in a classroom.

    - Poisson distribution helps predict the probability of certain events happening when you know how often that 
    event has occurred. 
      - It can be used by businessmen to make forecasts about the number of customers on certain days and allows 
      them to adjust supply according to the demand.

    - Exponential distribution is concerned with the amount of time until a specific event occurs. 
      - For example, how long a car battery would last, in months.
    
    
#### 2. What is OOB error and how does it occur? 
    - For each bootstrap sample, there is one-third of data that was not used in the creation of the tree, 
    i.e., it was out of the sample. This data is referred to as out of bag data. 
    - In order to get an unbiased measure of the accuracy of the model over test data, out of bag error is used. 
    - The out of bag data is passed for each tree is passed through that tree and the outputs are aggregated to 
    give out of bag error. 
    - This percentage error is quite effective in estimating the error in the testing set and does not require 
    further cross-validation. 

    
#### 3. Differentiate between univariate, bivariate, and multivariate analysis.
    - These are descriptive statistical analysis techniques that can be differentiated based on the number of 
    variables involved at a given point in time. For example, the pie charts of sales based on territory involve
    only one variable and can be referred to as univariate analysis.
    - If the analysis attempts to understand the difference between 2 variables at the time as in a scatterplot,
    then it is referred to as bivariate analysis. For example, analyzing the volume of sales and spending can 
    be considered as an example of bivariate analysis.
    - Analysis that deals with the study of more than two variables to understand the effect of variables on the 
    responses is referred to as multivariate analysis
    
#### 4. What are Interpolation and Extrapolation?
    - Estimating a value from 2 known values from a list of values is Interpolation. 
    - Extrapolation is approximating a value by extending a known set of values or facts.
    
    
#### 5. What is the difference between Cluster and Systematic Sampling?
    - Cluster sampling is a technique used when it becomes difficult to study the target population spread 
    across a wide area and simple random sampling cannot be applied. A cluster sample is a probability sample
    where each sampling unit is a collection or cluster of elements.
    - sampling is a statistical technique where elements are selected from an ordered sampling frame. 
    In systematic sampling, the list is progressed in a circular manner so once you reach the end of the list,
    it progresses from the top again. 
    The best example for systematic sampling is the equal probability method.
    
#### 6. Are expected value and mean value different?
    - They are not different but the terms are used in different contexts. Mean is generally referred to when
    talking about a probability distribution of sample population
    - whereas expected value is generally referred to in a random variable context.
    
    - For Sampling Data
        - The mean value is the only value that comes from the sampling data.
        - Expected Value is the mean of all the means i.e. the value that is built from multiple samples.
        The expected value is the population mean.
    - For Distributions
        - Mean value and Expected value are the same irrespective of the distribution, under the condition
        that the distribution is in the same population.
    
#### 7. Do gradient descent methods always converge to the same point?
    - No, they do not because in some cases it reaches a local minima or a local optima point. You don’t reach 
    the global optima point. 
    It depends on the data and starting conditions.
    
#### 8. You created a predictive model of a quantitative outcome variable using multiple regressions. What are the steps you would follow to validate the model?
    - Since the question asked is about post model building exercise, we will assume that you have already tested
    for null hypothesis, multicollinearity and Standard error of coefficients.
    - Once you have built the model, you should check for following –
        · Global F-test => to see the significance of group of independent variables on dependent variable
        · R^2
        · Adjusted R^2
        · RMSE, MAPE
    In addition to above mentioned quantitative metrics you should also check for-
        · Residual plot
        · Assumptions of linear regression 
    
#### 9. How can you deal with different types of seasonality in time series modelling?
    - Seasonality in time series occurs when time series shows a repeated pattern over time.
    E.g., stationary sales decrease during the holiday season, air conditioner sales increase during the summers
    etc. are few examples of seasonality in a time series. 
    - Seasonality makes your time series non-stationary because the average value of the variables at different 
    time periods. 
    
    **Differentiating a time series is generally known as the best method of removing seasonality from a time series.**
    Seasonal differencing can be defined as a numerical difference between a particular value and a value with a
    periodic lag 
    (i.e. 12, if monthly seasonality is present)
    
#### 10. Can you cite some examples where a false positive is more important than a false negative?
    - In the medical field, assume you have to give chemotherapy to patients. Your lab tests patients
    for certain vital information and based on those results they decide to give radiation therapy to a patient.
    Assume a patient comes to that hospital and he is tested positive for cancer (But he doesn’t have cancer) 
    based on lab prediction. What will happen to him? (Assuming Sensitivity is 1)
    
    - One more example might come from marketing. Let’s say an ecommerce company decided to give a $1000 Gift 
    voucher to the customers whom they assume to purchase at least $5000 worth of items. They send free voucher
    mail directly to 100 customers without any minimum purchase condition because they assume to make at least
    20% profit on sold items above 5K. 
    Now what if they have sent it to false positive cases? 
    
#### 11. Can you cite some examples where a false negative is more important than a false positive?
    - Assume there is an airport ‘A’ which has received high security threats and based on certain characteristics
    they identify whether a particular passenger can be a threat or not. Due to shortage of staff they decided to 
    scan passengers being predicted as risk positives by their predictive model.
    What will happen if a true threat customer is being flagged as non-threat by the airport model?

    - Another example can be the judicial system. What if Jury or judge decides to make a criminal go free?
    - What if you rejected to marry a very good person based on your predictive model and you happen to meet him/her
    after a few years and realize that you had a false negative?
    
#### 12. Can you cite some examples where both false positive and false negatives are equally important?
    - In the banking industry giving loans is the primary source of making money but at the same time if your repayment rate 
    is not good you will not make any profit, rather you will risk huge losses. Banks don’t want to lose good customers and 
    at the same point of time they don’t want to acquire bad customers. 
    In this scenario both the false positives and false negatives become very important to measure.

    - These days we hear many cases of players using steroids during sports competitions. Every player has to go through
    a steroid test before the game starts. A false positive can ruin the career of a Great sportsman and a false
    negative can make the game unfair.
    
    
#### 13. What do you understand about the statistical power of sensitivity and how do you calculate it?
    - Sensitivity is commonly used to validate the accuracy of a classifier (Logistic, SVM, RF etc.).
      Sensitivity is nothing but “Predicted TRUE events/ Total events”. True events here are the events which were
      true and the model also predicted them as true. 
      
    - Calculation of sensitivity is pretty straightforwardSensitivity = True Positives /Positives in Actual 
    Dependent Variable Where, True positives are Positive events which are correctly classified as Positives.
    
    
#### 14. What is the advantage of performing dimensionality reduction before fitting an SVM?
    - Support Vector Machine Learning Algorithm performs better in the reduced space. It is beneficial to perform
    dimensionality reduction before fitting an SVM if the number of features is large when compared to the number
    of observations.
    - In general, the curse of dimensionality happens, the higher the dimension of data, the lower then distance 
    between 2 points.
    [In fairly large dimension >100]
    
#### 15. How will you find the correlation between a categorical variable and a continuous variable ?
    - You can use the analysis of covariance technique to find the correlation between a categorical variable and 
    a continuous variable.
    
#### 16. What are the assumptions of Logistic Regression?
    - Logistic regression does not make many of the key assumptions of linear regression and general linear models
    that are based on ordinary least squares algorithms – particularly regarding linearity, normality, homoscedasticity,
    and measurement level.

    - First, logistic regression does not require a linear relationship between the dependent and independent variables.
    - Second, the error terms (residuals) do not need to be normally distributed.  
    - Third, homoscedasticity is not required.  
    - Finally, the dependent variable in logistic regression is not measured on an interval or ratio scale.

    However, some other assumptions still apply.
    
#### 17. What is the difference between PCA & SVD? 
    - PCA is Dimensionality Reduction Technique Where it is mathematically defined.
      Covariance Matrix S(n x n) => Square and Symatric Formed from **Eigen Vectors**
      Its Matrix decomposition will look like => S (dxd) = W (dxd) * Lamda (dxd) * W^T (dxd)
    - SVD X(n x d) is matrix factorization technique [Where matrix is not square but Rectangular in shape.]
      SVD X(nxd) = U (nxn) * Sigma (nxd) *V^T (dxd)
      Where Sigma is diagonal matrix with **Eigen values** in Diagonal places.
      
    
    
# Template:
#### 2. 
    - Answer
    
