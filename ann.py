#import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,  3:13].values
y = dataset.iloc[:, -1].values

#handling categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LE_X_1 = LabelEncoder()
LE_X_2 = LabelEncoder()
X[:, 1] = LE_X_1.fit_transform(X[:, 1])
X[:, 2] = LE_X_2.fit_transform(X[:, 2])
OHE_X_1 = OneHotEncoder(categorical_features= [1])
X = OHE_X_1.fit_transform(X).toarray()
X = X[:, 1:]
#splitting data into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#ANN

#import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#initialising the ANN
def build_classifier():
	classifier = Sequential()
	#adding the input layer and the first hidden layer and also DropOut for Dropout regularization
	classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
	classifier.add(Dropout(rate=0.1))
	#adding second hidden layer
	classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
	classifier.add(Dropout(rate=0.1))
	#adding the output layer
	classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
	#compiling the ANN
	classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return classifier

classifier = build_classifier()
#fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100);


#Evaluating the ANN, K-Fold CrossValidation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

k_classifier = KerasClassifier(build_fn=build_classifier,batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=k_classifier, X=X_train, y= y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
std = accuracies.std()

#Tuning the ANN using Grid Search Class
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def t_build_classifier():
	classifier = Sequential()
	#adding the input layer and the first hidden layer 
	classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
	#adding second hidden layer
	classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
	#adding the output layer
	classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
	#compiling the ANN
	classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return classifier
t_classifier = KerasClassifier(build_fn=t_build_classifier)
parameters = {'batch_size': [25, 32],
				'epochs':[100,500],
				'optimizer':['adam', 'rmsprop']
			}
grid_search = GridSearchCV(estimator=t_classifier,
							param_grid=parameters,
							scoring = 'accuracy',
							cv=10,
							n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_



# testing for a particular EMPLOYEE
# X_test_empl = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])
# X_test_empl[:, 1] = LE_X_1.transform(X_test_empl[:, 1])
# X_test_empl[:, 2] = LE_X_2.transform(X_test_empl[:, 2])
# X_test_empl = OHE_X_1.transform(X_test_empl).toarray()
# X_test_empl = X_test_empl[:, 1:]
# X_test_empl = sc.transform(X_test_empl)
# y_pred_empl = classifier.predict(X_test_empl)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
