#Importing The Libraries
import numpy as np
import pandas as pd 

#Importing the Dataset
dataset = pd.read_csv('corona.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,10].values

#Spliting the Dataset as TrainingSet and TestSet
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/5, random_state=0)

#Appling Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

#Impoting The Classifier 
from sklearn.svm import SVC
cls = SVC(kernel= 'rbf',random_state=0)
cls.fit(X_train, Y_train)

#Prediction of results for X_test
Y_pred = cls.predict(X_test)

#Calculation of confusion Matrix for correct prediction
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

#Taking Input form user for prediction 
spt = []
spt.append(int(input('Fever in fehrenheit:\n ')))
spt.append(int(input('Dry cough : (Yes = 1 or No=0)\n')))
spt.append(int(input('Difficulty in breathing :(Yes = 1 or mild = 0 or No= -1)\n')))
spt.append(int(input('RunnyNose : (Yes = 1 or No=0)\n')))
spt.append(int(input('Tiredness :(Yes = 1 or No=0)\n')))
spt.append(int(input('BodyAches : (Yes = 1 or mild = 0 or No= -1)\n')))
spt.append(int(input('Sore Throat : (Yes = 1 or mild = 0 or No= -1)\n')))
spt.append(int(input('Bluish lips or Face :(Yes = 1 or No=0)\n')))
spt.append(int(input('Persistent pain or pressure in the Chest :(Yes = 1 or mild = 0 or No= -1)\n')))
spt.append(int(input('Age : ')))

#Results of Input
prob_infection =cls.predict_proba([spt])
print ("Probability tHat you are CORONA POSITIVE :" , prob_infection[:,[1]])
print ("Probability tHat you are CORONA NEGATIVE :" , prob_infection[:,[0]])


#converting List in to 2D array
spt = np.asarray([spt])
numpyarray = spt

#tabular Form of symptoms status of patient
df = pd.DataFrame({'Age' : numpyarray[:, 9],'Pain_in_chest' : numpyarray[:, 8],'Bluish_lips' : numpyarray[:, 7],'SoreThroat' : numpyarray[:, 6],'BodyAches' : numpyarray[:, 5],'Tiredness' : numpyarray[:, 4],'RunnyNose' : numpyarray[:, 3],'Diff_in_Breath' : numpyarray[:, 2], 'Dry_cough': numpyarray[:,1], 'Fever' : numpyarray[:, 0]})
print(df)
