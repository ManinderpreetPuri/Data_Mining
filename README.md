# Data_Mining

Project 1

In this project, I build a data mining model using decision tree to predict wine quality based on physicochemical tests. 

▪ Dataset:
part1_data. 
 
o The first six columns are attributes of wine samples, and the last column is the class label denoting the wine quality. 
i. The wine sample attributes, including alcohol, density and chlorides, are all numeric values.
ii. Quality (class label) is of the categorial type where the value can be 5 or 7 denoting wine of average or excellent qualities respectively. 

▪ Requirements:
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


Project 2

Part 1

In this project, I build a neural network classifier (NN) for the given dataset. I worked with a subject/unit dataset taken from a major American university. This University is selective and therefore attracts a student body with relatively high entrance qualifications. The dataset has been partially cleaned but still contains blank or null values. I built a NN predictive stream that predicts At_Risk status, which is a logistic/binary/flag classification target. 

I answered the following questions:

1. Is the original data ready to be used by the Orange3 neural network classifier?
i. If not, state the reason and write a Python script to perform any necessary pre-processing so that the data becomes suitable to be used.
ii. Briefly describe the pre-processing carried out with a brief comment in python script and word document.
iii. Submit the pre-processed data in CSV format.

2. Create a NN classifier. The following parameters need to be defined.
o Number of hidden layers: ? 
o Number of neurons: ? 
o Maximum number of iterations: ? 
Use default settings for other parameters. Perform 10-fold cross-validation to evaluate the performance of the NN classifier with this data. The answer should include the followings: 
i. Python source codes reading the source data, building the learner, and performing 10-fold cross-validation.
ii. Accuracy and area under the receiver operating characteristic (ROC) curve (AUC). 
iii. What is your inference? Is NN the best classifier for this data?

3. Use the same dataset and build a NN classifier using PANDAS and SciKitLearn’s MLP 
Classifier.
i. Read the data set as a data frame object, pre-process the data and split it into 
training and testing dataset. (70% training and 30% testing) 
ii. Create an MLP classifier using the following parameters: 
o Number of hidden layers: 3 
o Number of neurons: 10 
o Maximum number of iterations: 6000 
iii. Evaluate the model, get predictions, and generate a confusion matrix. Explain it in a word document 


Part 2

In this project, I applied K-means clustering on a set of signal data from a phased array of 16 high-frequency antennas. This (built-in) Orange3 dataset can be loaded with the Python statement: 

data_tab = Table('ionosphere')

The data is then stored in the Orange3 Table object data_tab. and the class label “y” indicates whether a signal pass through the ionosphere (shown as “b” in y) or present some types of structure in the ionosphere (shown as “g” in y). All other table columns with their names starting with “a” are signal readings. 
a. Cluster the data using the scikit-learn K-means clustering with K = 2 and specifying 
random_state = 0. Submit Python script file for this process.
b. Use the clustering results obtained from Part II (a) and the class label “y” to count and fill in the number of signals for each of the four categories in the table below. 
Use Python to perform the calculations. 
