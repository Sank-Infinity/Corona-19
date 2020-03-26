# Corona-19
1) Python Code for Probability prediction for corona disease for patient using machine learning model.  

2) This Machine learning model based on RANDOM DATA set of BASIC SYMPTOMS like Fever, 	Dry cough,	Difficulty in breathing, RunnyNose,	Tiredness,	Body Aches,	Sore throat,	Bluish lips or face,	Persistent pain or pressure in the chest and Age.

3) Change the data set and index accordingly and train the model. you will get probability whether, patient is positive or negative.
4) If Probability of positiveness is more than 0.5 than patient might be suffer from corona otherwise not.

# GuideLines

-Make sure You have installed all depedencies required for script.

-For getting well organized results use SPYDER IDLE.

-Run script unitwise. 

-confusion matrix will return a 2x2 matrix. where,addition A11 + A22 represents correct predictions and addition of A12+A21 represents mismatched values of predictions and real values . 

# NOTE
In this model SupporVectorClassifier from scikit learn library is used because our result is in the format of (yes or no) 0 or 1 which means it returns bullean value. 

You can also go for other classifiers such as LogisticRegressionClassifier, DecisionTreeClassifier, RandomForestClassifier etc.
