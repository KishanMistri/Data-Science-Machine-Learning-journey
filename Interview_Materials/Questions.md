##### 1. You are given a train data set having 1000 columns and 1 million rows. The data set is based on a classification problem. Your manager has asked you to reduce the dimension of this data so that model computation time can be reduced. Your machine has memory constraints. What would you do? (You are free to make practical assumptions.)
   - Processing a high dimensional data on a limited memory machine is a strenuous task, your interviewer would be fully aware of that. Following are the methods you can use to tackle such situation:
   - Since we have lower RAM, we should close all other applications in our machine, including the web browser, so that most of the memory can be put to use.
   - We can randomly sample the data set. This means, we can create a smaller data set, let's say, having 1000 variables and 300000 rows and do the computations.
   - To reduce dimensionality, we can separate the numerical and categorical variables and remove the correlated variables. For numerical variables, we'll use correlation. For categorical variables, we'll use chi-square test.
   - Also, we can use **[PCA](http://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/)** and pick the components which can explain the maximum variance in the data set.
   - We can also apply our business understanding to estimate which all predictors can impact the response variable. But, this is an intuitive approach, failing to identify useful predictors might result in significant loss of information.




Question_Template
##### X. Question
   - Answer

