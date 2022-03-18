#ASSIGNMENT PART A:


#Importing Pandas library
import pandas as pd
import numpy as np

#Reading file into data frame
csv_path = 'winequality-white-v3.csv'
df0 = pd.read_csv(csv_path) 

#Filtering missing values
missingchlorides = pd.isna(df0["chlorides"])
missingIndices = df0[missingchlorides].index

#Replacing missing values by mean
meanchlorides = round(df0["chlorides"].mean(), 3)
df0["chlorides"].where(~ missingchlorides, meanchlorides, inplace=True)

#Deleteing missing values at the EOF
missing_winequailty = pd.isna(df0["alcohol"])
missingIndices = df0[missing_winequailty].index
df1 = df0.drop(missingIndices, axis=0)

#Transformaing the quality variable to 0 and 1
def transformQuality(quality):
    if quality == 5:
        quality = 0
    else:
        quality = 1
    return quality

df1["quality"] = df1["quality"].apply(transformQuality)

#Saving into a csv file
df1.to_csv('filtered-winequality-white-v3.csv') 


#ASSIGNMENT PART B:

#PART i:
    
#Importing Orange Library
#Importing “SklTreeLearner”, cross validation, scoring and confusion matrix
from Orange.data import Table, Domain
from Orange.classification import SklTreeLearner
from Orange.evaluation import CrossValidation, scoring
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

#Reading the fileterd data
Filtered_data = Table.from_file('filtered-winequality-white-v3.csv')

#Defining features
feature_vars = list(Filtered_data.domain.variables[1:6])
class_label_var = Filtered_data.domain.variables[7]

#Defining domain
winequality_domain = Domain(feature_vars, class_label_var)
Filtered_data= Table.from_table(domain=winequality_domain, source=Filtered_data)

#Shuffling and splitting data for traning and testing
Filtered_data.shuffle()
train_data_tab = Filtered_data[:1800]
test_data_tab = Filtered_data[1800:]

#Creating tree learner and decision tree
tree_learner = SklTreeLearner()
decision_tree = tree_learner(train_data_tab)

#Creating external prediction function having the decision tree and the input  
def decision_tree_predict(d_tree, input_data):
    predicted_label_vals = d_tree(input_data)

    predicted_labels = []
    for val in predicted_label_vals:
        predicted_labels.append(input_data.domain.class_var.values[int(val)])

    return predicted_labels


#PART ii:
    
#Estimating the accuracy of decision_tree using the entire testing data
#Performance estimation by comparing the predicted class with the actual class 
predicted_class_labels = decision_tree_predict(decision_tree, test_data_tab[:])

num_of_test_samples = len(predicted_class_labels)
num_of_correct_predictions = 0

for i in range(num_of_test_samples):
    if predicted_class_labels[i] == test_data_tab[i, test_data_tab.domain.class_var]:
        num_of_correct_predictions += 1

accuracy = num_of_correct_predictions / num_of_test_samples
print('Accuracy = {:.3f}'.format(accuracy))

#Evaluating the performance of a decision tree classifier using cross-validation
eval_results = CrossValidation(Filtered_data, [tree_learner], k=10)
print("Accuracy: {:.3f}".format(scoring.CA(eval_results)[0]))
print("AUC: {:.3f}".format(scoring.AUC(eval_results)[0]))

#Confusion Matrix
y_true = test_data_tab[:, 5].astype(int).astype(str)
y_pred = predicted_class_labels
print(y_true)
print(y_pred)
confusion_matrix= scoring.confusion_matrix(y_true, y_pred)
print(confusion_matrix)