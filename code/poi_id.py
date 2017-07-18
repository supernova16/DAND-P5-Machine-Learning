#!/usr/bin/python


import pickle
import numpy as np
import pandas as pd

# utilities
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# pre-processing
from sklearn.preprocessing import MinMaxScaler

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from tester import dump_classifier_and_data

"""
Data Preparation
"""

# load the data dict and convert to pandas
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame(data_dict)
df = df.T

# move 'poi' column to head of list. this will be the label
cols = df.columns.tolist()
cols.insert(0, cols.pop(cols.index('poi')))
df = df.reindex(columns = cols)

### Task 1: Select what features you'll use.
### Task 2: Remove outliers
### Task 3: Create new feature(s) and  feature processing

# NaN and outliers
df.replace('NaN', np.nan, inplace = True)
df = df.fillna(0)
df = df.drop(['LOCKHART EUGENE E','TOTAL','THE TRAVEL AGENCY IN THE PARK'], axis=0)


"""
Feature Engineering
"""
# remove useless feature
df = df.drop(['email_address'], axis=1)

# create new features
df['from_poi_rate'] = df['from_poi_to_this_person'] / df['to_messages']
df['from_poi_rate'] = df['from_poi_rate'].fillna(0)

df['sent_to_poi_rate'] = df['from_this_person_to_poi'] / df['from_messages']
df['sent_to_poi_rate'] = df['sent_to_poi_rate'].fillna(0)

df['shared_with_poi_rate'] = df['shared_receipt_with_poi'] / df['to_messages']
df['shared_with_poi_rate'] = df['shared_with_poi_rate'].fillna(0)

# log-scaling for original features

for col in df.columns[1:]:
    df[col] = [np.log(abs(v)) if v != 0 else 0 for v in df[col]]


df_features = df.drop('poi', axis=1)
df_labels = df['poi']



### Task 4: Try a varity of classifiers

algorithms = [
    GaussianNB(),
    LogisticRegression(class_weight='balanced', random_state=42),
    make_pipeline(MinMaxScaler(), SVC(class_weight='balanced', random_state=42)),
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    RandomForestClassifier(class_weight='balanced', random_state=42),
    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(class_weight='balanced'), random_state=42)
]


# list containing, for each algorithm above, a set of parameters with possible values to try and compare
params = [
    { },  # GaussianNB, no parameters to try
    {  # LogisticRegression
        'penalty' : ('l1', 'l2'),
        'C' : [0.1, 1.0, 10, 100, 1000, 10000],
        'max_iter' : [10, 50, 100, 150, 200]
    },
    {  # SVC pipeline
        'svc__kernel' : ('linear', 'rbf', 'poly', 'sigmoid'),
        'svc__C' : [0.1, 1.0, 10, 100, 1000, 10000],
        'svc__gamma' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]
    },
    {  # DecisionTreeClassifier
        'criterion' : ('gini', 'entropy'),
        'splitter' : ('best', 'random'),
        'min_samples_split' : [2, 5, 10, 20, 40, 70]
    },
    {  # RandomForestClassifier
        'n_estimators' : [5, 10, 50],
        'criterion' : ('gini', 'entropy'),
        'min_samples_split' : [2, 5, 10, 20, 40, 70]
    },
    {  # AdaBoostClassifier
        'n_estimators' : [5, 10, 50],
        'learning_rate' : [0.001, 0.01, 0.1, 1.0]
    }
]

best_estimator = None  # will contain final best algorithm
best_score = 0  # will contain final estimated best performance score

# perform actual iteration and testing (GridSearch)
for ii, algorithm in enumerate(algorithms):
    # GridSearch also performs CrossValidation to automatically generate training and testing sets of data
    grid_search = GridSearchCV(algorithm, params[ii],cv=5)
    grid_search.fit(df_features, df_labels)

    best_estimator_ii = grid_search.best_estimator_
    best_score_ii = grid_search.best_score_

    if best_score_ii > best_score:
        best_estimator = best_estimator_ii
        best_score = best_score_ii



clf = best_estimator  # store for later
print('best estimator is: \n',clf)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import cross_val_predict

# evaluate model using cross validation
predictions = cross_val_predict(clf, df_features, df_labels, cv=5)  # StratifiedKFold

# interesting metrics in this case are Recall and Precision.
# Recall is the number of true POIs that were included in our predicted POIs
# Precision is the number of true positives (true POIs) in our predicted POIs
# Additionally, the F1 score combines both Recall and Precision into a metric that varies from 0 (worst) to 1 (best)

from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score

print('accuracy  :', accuracy_score(df_labels, predictions))
print('f1        :', f1_score(df_labels, predictions))
print('precision :', precision_score(df_labels, predictions))
print('recall    :', recall_score(df_labels, predictions))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, df, df_features)