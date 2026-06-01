# IMPORTING THE DEPENDENCIES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# DATA COLLECTION AND ANALYSIS
# Loading the data from csv file to pandas Dataframe
Parkinsons_Data = pd.read_csv("parkinsons.csv")
# Printing the first five lines(First 5 by Default)
# print(Parkinsons_Data.head())

# Printing the total number of rows and columns
# print(Parkinsons_Data.shape)

# Prints some more details
# print(Parkinsons_Data.info())

# Prints the STATISTICAL DESCRIPTION
# print(Parkinsons_Data.describe())

# IN OUR CASE STATUS VARIABLE IS OUR TARGET VARIABLE
# Distribution of Target Variable(Status)
# 1=> People with Parkinson's Disease
# 0=> Healthy People
# print(Parkinsons_Data['status'].value_counts())


# Grouping the data based on the target variable
Parkinsons_Data.groupby('status').mean()

# Separating features and targets
X = Parkinsons_Data.drop(columns=['name' , 'status'], axis=1)
Y = Parkinsons_Data['status']

# Splitting the Data into Training Data and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 , random_state=2)
# print(X.shape, X_train.shape, X_test.shape)
# print(X_train,Y_train)

# Data Standardization
scaler = StandardScaler()
# Training our model
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# MODEL TRAINING
# SUPPORT VECTOR MACHINE MODEL
model = svm.SVC(kernel='linear')
# Training the SVM Model
model.fit(X_train, Y_train)

# EVALUATION
# Accuracy score of X_train data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("Accuracy Score of the Training Data is: ",training_data_accuracy)

# Accuracy Score of Test Data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy Score of the Test Data is: ", test_data_accuracy)

# BUILDING A PREDICTIVE SYSTEM
# MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP,MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE
report_results = input("Enter the report details seperated by comma:")
list1 = report_results.split(',')
list2 = map(float,list1)
input_data = tuple(list2)


# CHANGING INPUT DATA TO NUMPY ARRAY
input_data_as_numpy_array = np.asarray(input_data)

# RESHAPE THE NUMPY ARRAY
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# STANDARDISE THE DATA
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print("STATUS:",prediction)

if prediction[0]==0:
    print("The person is HEALTHY and is not having any symptoms of PARKINSON'S DISEASE")
else:
    print("The person is having PARKINSON'S DISEASE")

