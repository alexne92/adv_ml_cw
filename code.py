# Import basic libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('train.csv')
dataset = dataset[['text','author']]

# Clean the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
# for i in range(0, len(dataset['text'])):
for i in range(0, 100): # sample of 100 data
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Find top-10 words
words = []
for j in range(0,len(corpus)):
    for k in range(0,len(corpus[j].split())):
        words.append(corpus[j].split()[k])
print(nltk.FreqDist(words).most_common(10))

# Create Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

# Find information for each word
df = pd.DataFrame(X, columns = cv.get_feature_names())
df.sum(axis = 0).sort_values().tail()
df.sum(axis = 0).sort_values().tail(10) # for top-10

# Else
X.sum(axis = 0) # not practical

# Label the target output
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y_labelncoder = LabelEncoder()
y = y_labelncoder.fit_transform(dataset['author'])
y = y[:100] # Sample of the target output
from keras.utils import np_utils
y_target = np_utils.to_categorical(y)

# Split dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X, y_target, test_size = 0.20, random_state = 0)

# Fit Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred_nb = nb_classifier.predict(X_test)

# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)

# Fit Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 0)
lr_classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred_lr = lr_classifier.predict(X_test)

# Make the Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Fit K-NN
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p = 2) # minkowski with p = 2 refers to the Euclidean distance
knn_classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred_knn = knn_classifier.predict(X_test)

# Make the Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

from sklearn.model_selection import GridSearchCV

# Tune the number of neighbors (n_neighbors = 1)
parameters = [{'n_neighbors': range(1,25)}]
grid_search = GridSearchCV(estimator = knn_classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_

# Fit SVM to the Training set
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'linear', random_state = 0)
svm_classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred_svm = svm_classifier.predict(X_test)

# Make the Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)

# Fit Kernel SVM-RBF
from sklearn.svm import SVC
rbf_classifier = SVC(kernel = 'rbf', random_state = 0)
rbf_classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred_rbf = rbf_classifier.predict(X_test)

# Make the Confusion Matrix
cm_rbf = confusion_matrix(y_test, y_pred_rbf)

# Fit Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0) # entropy for information gain (gini for the Gini impurity)
dt_classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred_dt = dt_classifier.predict(X_test)

# Make the Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)

# Find best criterion (criterion = 'entropy')
parameters_dt = [{'criterion': ['entropy', 'gini']}]
grid_search_dt = GridSearchCV(estimator = dt_classifier,
                           param_grid = parameters_dt,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search_dt = grid_search_dt.fit(X_train, y_train)
best_parameters_dt = grid_search_dt.best_params_


# Fit Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 301, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred_rf = rf_classifier.predict(X_test)

# Make the Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Tune the number of the trees and find best criterion (criterion = 'entropy', n_estimators = 301)
parameters_rf = [{'n_estimators' : range(1,1000,100), 'criterion' : ['entropy', 'gini']}]
grid_search_rf = GridSearchCV(estimator = rf_classifier,
                           param_grid = parameters_rf,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_rf = grid_search_rf.fit(X_train, y_train)
best_parameters_dt = grid_search_rf.best_params_


# Import Keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding


# Initialise the Neural Network
nn_classifier = Sequential()

# Add input layer and first hidden layer

non = round(np.mean([X_train_nn.shape[1], y_train_nn.shape[1]])) # Number of nodes for the hidden layer
nn_classifier.add(Dense(output_dim = int(non), 
                        init = 'uniform', 
                        activation = 'relu', 
                        input_dim = X_train.shape[1]))

# Add second hidden layer
nn_classifier.add(Dense(output_dim = int(round(0.5*non)), 
                        init = 'uniform', 
                        activation = 'relu'))

# Add output layer
nn_classifier.add(Dense(output_dim = y_train_nn.shape[1], 
                        init = 'uniform', 
                        activation = 'softmax'))

# Compile the Neural Network
nn_classifier.compile(optimizer = 'adam', 
                      loss = 'categorical_crossentropy', 
                      metrics = ['accuracy'])

# Fit the Neural Network
nn_classifier.fit(X_train_nn, y_train_nn, batch_size = 10, nb_epoch = 50)

# Predict the Test set results
y_pred_nn = nn_classifier.predict(X_test)
result = (y_pred_nn == y_pred_nn.max(axis = 1, keepdims = 1)).astype(float)
y_pred_ann = pd.DataFrame(result, columns = [0, 1, 2]).idxmax(1).as_matrix()

# Make the Confusion Matrix
cm_nn = confusion_matrix(y_test, y_pred_ann)


# Initialise the Neural Network with LSTM
lstm_classifier = Sequential()

# Add input layer and first hidden layer

non = round(np.mean([X_train_nn.shape[1], y_train_nn.shape[1]])) # Number of nodes for the hidden layer
lstm_classifier.add(Embedding(500, 
                              40, 
                              input_length = X_train.shape[1]))

# Add second hidden layer
lstm_classifier.add(Conv1D(filters= 32, 
                           kernel_size=3, 
                           padding='same', 
                           activation='relu'))

lstm_classifier.add(MaxPooling1D(pool_size=2))

lstm_classifier.add(LSTM(100))

# Add output layer
lstm_classifier.add(Dense(output_dim = y_train_nn.shape[1], 
                          init = 'uniform', 
                          activation = 'softmax'))

# Compile the Neural Network
lstm_classifier.compile(optimizer = 'adam', 
                        loss = 'categorical_crossentropy', 
                        metrics = ['accuracy'])

# Fit the Neural Network

lstm_classifier.fit(X_train_nn, 
                    y_train_nn, 
                    batch_size = 64, 
                    epochs = 5)

# Predict the Test set results
y_pred_lstm = lstm_classifier.predict(X_test)
result_lstm = (y_pred_lstm == y_pred_lstm.max(axis = 1, keepdims = 1)).astype(float)
y_pred_ann_lstm = pd.DataFrame(result_lstm, columns = [0, 1, 2]).idxmax(1).as_matrix()

# Make the Confusion Matrix
cm_lstm = confusion_matrix(y_test, y_pred_ann_lstm)

# Fit XGBoost
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred_xgb = xgb_classifier.predict(X_test)

# Make the Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

#Store the predictions
y_preds = [y_pred_nb, 
           y_pred_lr, 
           y_pred_knn, 
           y_pred_svm, 
           y_pred_rbf, 
           y_pred_dt, 
           y_pred_rf, 
           y_pred_ann,
           y_pred_ann_lstm,
           y_pred_xgb]

# Find metrics of classification
accuracy = []
precision = []
recall = []
f_one_score = []
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

for y_pred in y_preds:
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred, average = 'macro'))
    recall.append(recall_score(y_test, y_pred, average = 'macro'))
    f_one_score.append(f1_score(y_test, y_pred, average= 'macro'))
    
index = ["Naive Bayes", 
         "Logistic Regression",
         "K Nearest Neighbour",
         "Support Vector Machine",
         "Kernel Support Vector Machine",
         "Decision Tree",
         "Random Forest",
         "Neural Network",
         "Recurrent Neural Network",
         "XGBoost"]    

metrics = pd.DataFrame({'accuracy' : pd.Series(accuracy, index = index),
                        'precision' : pd.Series(precision, index = index),
                        'recall' : pd.Series(recall, index = index),
                        'f1_score' : pd.Series(f_one_score, index = index)})
    
metrics = metrics[['accuracy', 'precision', 'recall', 'f1_score']]    

print(metrics)

# Make cross-validation

accuracy_mean = []
accuracy_std = []
precision_mean = []
precision_std = []
recall_mean = []
recall_std = []
f_one_score_mean = []
f_one_score_std = []
fit_time = []

classifiers = [nb_classifier,
               lr_classifier,
               knn_classifier,
               svm_classifier,
               rbf_classifier,
               dt_classifier,
               rf_classifier,
               nn_classifier,
               lstm_classifier,
               xgb_classifier]

from sklearn.model_selection import cross_validate

for classifier in classifiers:
    scores = cross_validate(estimator = classifier, 
							X = X, 
							y = y, 
							cv = 10,
							scoring = ['accuracy','precision_macro', 'recall_macro', 'f1_macro'])
    accuracy_mean.append(scores['test_accuracy'].mean())
    accuracy_std.append(scores['test_accuracy'].std())
    precision_mean.append(scores['test_precision_macro'].mean())
    precision_std.append(scores['test_precision_macro'].std())
    recall_mean.append(scores['test_recall_macro'].mean())
    recall_std.append(scores['test_recall_macro'].std())
    f_one_score_mean.append(scores['test_f1_macro'].mean())
    f_one_score_std.append(scores['test_f1_macro'].std())
    fit_time.append(scores['fit_time'].mean())

metrics2 = pd.DataFrame({'accuracy_mean' : pd.Series(accuracy_mean, index = index),
                        'accuracy_std' : pd.Series(accuracy_std, index = index),
                        'precision_mean' : pd.Series(precision_mean, index = index),
                        'precision_std' : pd.Series(precision_std, index = index),
                        'recall_mean' : pd.Series(recall_mean, index = index),
                        'recall_std' : pd.Series(recall_std, index = index),
                        'f1_score_mean' : pd.Series(f_one_score_mean, index = index),
                        'f1_score_std' : pd.Series(f_one_score_std, index = index),
                        'fit_time' : pd.Series(fit_time, index = index)})
  
metrics2 = metrics2[['accuracy_mean',
                     'accuracy_std',
                     'precision_mean',
                     'precision_std',
                     'recall_mean',
                     'recall_std',
                     'f1_score_mean',
                     'f1_score_std',
                     'fit_time']]    

print(metrics2)