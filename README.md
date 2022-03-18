# Data_Mining

Project 1

In this project, I build a data mining model using decision tree to predict wine quality based on physicochemical tests. 

▪ Dataset
part1_data. 
 
o The first six columns are attributes of wine samples, and the last column is the class label denoting the wine quality. 
i. The wine sample attributes, including alcohol, density and chlorides, are all numeric values.
ii. Quality (class label) is of the categorial type where the value can be 5 or 7 denoting wine of average or excellent qualities respectively. 

▪ Requirements. 
I answered the following questions:
a. Data pre-processing 
There are one or more attributes with missing values.
o Write a Python script to read the input CSV file, replace the missing values with the mean value for that attribute, and then export the entire dataset to another 
CSV file. 
o Briefly describe the pre-processing you performed in your final report. 

b. Wine quality prediction 
Using the data exported in part (a), create an Orange3 decision tree learner, and perform 10-fold cross-validation to evaluate the performance of decision tree 
classifier with this data. To answer question b, need to provide
i. Python source code reading the source data, building the learner, and performing 10-fold cross-validation.
ii. Performance evaluation results including confusion matrix, accuracy and area under (receiver operating characteristic, ROC) curve (AUC). 

c. Open discussion
There is an argument that not all the given attributes contribute to the wine quality prediction results by decision tree. Do you agree with this claim? Justify your 
answer with both explanation and computational demonstration in your final report.
