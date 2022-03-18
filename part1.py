#ASSIGNMENT PART I:

#1. 

#Importing Pandas library
import pandas as pd

#Reading file into data frame
csv_path = 'USA_Uni_Subject .csv'
df0 = pd.read_csv(csv_path)

#Number of missing values in data set
before = pd.isna(df0).sum()
print(before)

#Filtering missing values
missingHSGPA = pd.isna(df0["HSGPA"])
missingLAST_ACT_ENGL_SCORE = pd.isna(df0["LAST_ACT_ENGL_SCORE"])
missingLAST_ACT_MATH_SCORE = pd.isna(df0["LAST_ACT_MATH_SCORE"])
missingLAST_ACT_READ_SCORE = pd.isna(df0["LAST_ACT_READ_SCORE"])
missingLAST_ACT_SCIRE_SCORE = pd.isna(df0["LAST_ACT_SCIRE_SCORE"])
missingLAST_ACT_COMP_SCORE = pd.isna(df0["LAST_ACT_COMP_SCORE"])

#Replacing all missing values by mean
meanHSGPA = round(df0["HSGPA"].mean(), 4)
df0["HSGPA"].where(~ missingHSGPA, meanHSGPA, inplace=True)

meanLAST_ACT_ENGL_SCORE = round(df0["LAST_ACT_ENGL_SCORE"].mean(), 4)
df0["LAST_ACT_ENGL_SCORE"].where(~ missingLAST_ACT_ENGL_SCORE, meanLAST_ACT_ENGL_SCORE, inplace=True)

meanLAST_ACT_MATH_SCORE = round(df0["LAST_ACT_MATH_SCORE"].mean(), 4)
df0["LAST_ACT_MATH_SCORE"].where(~ missingLAST_ACT_MATH_SCORE, meanLAST_ACT_MATH_SCORE, inplace=True)

meanLAST_ACT_READ_SCORE = round(df0["LAST_ACT_READ_SCORE"].mean(), 4)
df0["LAST_ACT_READ_SCORE"].where(~ missingLAST_ACT_READ_SCORE, meanLAST_ACT_READ_SCORE, inplace=True)

meanLAST_ACT_SCIRE_SCORE = round(df0["LAST_ACT_MATH_SCORE"].mean(), 4)
df0["LAST_ACT_SCIRE_SCORE"].where(~ missingLAST_ACT_SCIRE_SCORE, meanLAST_ACT_SCIRE_SCORE, inplace=True)

meanLAST_ACT_COMP_SCORE = round(df0["LAST_ACT_COMP_SCORE"].mean(), 4)
df0["LAST_ACT_COMP_SCORE"].where(~ missingLAST_ACT_COMP_SCORE, meanLAST_ACT_COMP_SCORE, inplace=True)

#Number of missing values after preprocessing
after = pd.isna(df0).sum()
print(after)

#Defining Function to Transform the SEX variable to 0 for M, 1 for F, 2 for ?, 3 for numeric values 
def transformSEX(SEX):
    if SEX == 'M':
        SEX = 0
    elif SEX =='F':
        SEX = 1
    elif SEX =='?':
        SEX = 2
    else:
        SEX = 3
    return SEX

#Applying Transformation to data set
df0["SEX"] = df0["SEX"].apply(transformSEX)

#Filtering out unwanted data: 2 for ?, 3 for numeric
indexNames = df0[df0['SEX'] == 2].index
indexNames1 = df0[df0['SEX'] == 3].index
 
# Delete these row indexes from dataFrame
df0.drop(indexNames , inplace=True)
df0.drop(indexNames1 , inplace=True)

#Function to Transform At_risk variable to 0 and 1
def transformAt_Risk(At_Risk):
    if At_Risk == 'F':
        At_Risk = 0
    else:
        At_Risk = 1
    return At_Risk

#Applying Transformation to data set
df0["At_Risk"] = df0["At_Risk"].apply(transformAt_Risk)

#Rearranging columns to bring 'At_Risk' to one side for ease of use
df0 = df0[['GRD_PTS_PER_UNIT', 'CATALOG_NBR', 'GPAO', 'ANON_INSTR_ID', 'TERM','HSGPA','LAST_ACT_ENGL_SCORE', 'LAST_ACT_MATH_SCORE', 'LAST_ACT_READ_SCORE', 'LAST_ACT_SCIRE_SCORE', 'LAST_ACT_COMP_SCORE', 'SEX', 'At_Risk']]
     
#Saving into a csv file
df0.to_csv('Preprocessed_USA_Uni_Subject.csv')




#2.

#Importing Orange libraries
from Orange.classification import NNClassificationLearner
from Orange.data import Table
from Orange.data import Domain
from Orange.evaluation import CrossValidation, scoring

#Loading the dataset to a Table object
data_tab = Table('Preprocessed_USA_Uni_Subject')

#Defining features ,class variable and domain
feature_var = list(data_tab.domain.variables[1:12])
class_label_var = data_tab.domain.variables[13]
USA_Uni_Subject_domain = Domain(feature_var, class_label_var)
data_tab = Table.from_table(domain=USA_Uni_Subject_domain, source=data_tab)

#Building the ANN classifier
nn_learner = NNClassificationLearner(hidden_layer_sizes=(10,10), max_iter=2000)

#Evaluating results with 10-fold cross-validation
eval_results = CrossValidation(data_tab, [nn_learner], k=10)
print("Accuracy: {:.3f}".format(scoring.CA(eval_results)[0]))
print("AUC: {:.3f}".format(scoring.AUC(eval_results)[0]))



#3.

#Importing Pandas and SciKitLearn library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Reading the file
USA_Uni_Subject_data = pd.read_csv('Preprocessed_USA_Uni_Subject.csv', index_col = 0)

#Spliting data into class and feature variables
X= USA_Uni_Subject_data.drop('At_Risk' , axis= 1)
Y = USA_Uni_Subject_data['At_Risk']

# Splitting data into Training(70%) and Testing(30%) 
X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.30)

#Data Preprocessing
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Building the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=6000)
mlp.fit(X_train, Y_train)

#Evaluating results and generating a confusion matrix
predictions = mlp.predict(X_test)
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test,predictions))

