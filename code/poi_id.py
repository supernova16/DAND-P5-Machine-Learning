#!/usr/bin/python3.6


import pickle
import numpy as np
import pandas as pd
import warnings

# cross validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

# pre-processing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# evaluation
from tester import dump_classifier_and_data, load_classifier_and_data, test_classifier
from sklearn.model_selection import cross_val_predict

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

def data_info():
    print('Number of persons:', len(df.index))
    print('Number of features:', len(df.columns))
    print('Number of data points:', len(df.index) * len(df.columns))
    print('Number of POIs:', len(df[df['poi'] == True]))
    print('Number of non-POIs:', len(df[df['poi'] == False]))


print('\n\n==============\nOriginal Dataset Info:\n\n',df.columns)
data_info()

### Task 1: Select what features you'll use.
### Task 2: Remove outliers
### Task 3: Create new feature(s) and  feature processing

"""
Feature Engineering
"""

# count NaN numbers
df.replace('NaN', np.nan, inplace = True)
features_nan_count = df.isnull().sum(axis = 0).sort_values(ascending=False)
person_nan_count = df.isnull().sum(axis = 1).sort_values(ascending=False)

print('\n\n==============\nTop 10 features NaN:\n\n',features_nan_count[:10])
print('\nTop 10 person NaN:\n\n',person_nan_count[:10])

# process NaN and remove outliers
df = df.fillna(0)
df = df.drop(['LOCKHART EUGENE E','TOTAL','THE TRAVEL AGENCY IN THE PARK'], axis=0)

print('\n\n==============\nProcessed Dataset Info:\n\n',df.columns)
data_info()


# remove useless feature
df = df.drop(['email_address'], axis=1) # email address has no useful info

# create new features
df['from_poi_rate'] = df['from_poi_to_this_person'] / df['to_messages']
df['from_poi_rate'] = df['from_poi_rate'].fillna(0)

df['sent_to_poi_rate'] = df['from_this_person_to_poi'] / df['from_messages']
df['sent_to_poi_rate'] = df['sent_to_poi_rate'].fillna(0)

df['shared_with_poi_rate'] = df['shared_receipt_with_poi'] / df['to_messages']
df['shared_with_poi_rate'] = df['shared_with_poi_rate'].fillna(0)

df_features = df.drop('poi', axis=1)
df_labels = df['poi']



# feature selection-KBest
k_best = SelectKBest(k='all')
k_best.fit(df_features, df_labels)

print('\n\n==============\nFeatures KBest Score:\n',pd.DataFrame(
    k_best.scores_,index=df_features.columns.tolist(),columns=['score']).sort_values('score', ascending=False))


# log-scaling for all original features
for col in df.columns[1:]:
    df[col] = [np.log(abs(v)) if v != 0 else 0 for v in df[col]]



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



"""
Hyperparameter with GridSearchCV 
"""

df_features = df.drop('poi', axis=1)
df_labels = df['poi']


algorithms = [
    GaussianNB(),
    LogisticRegression(class_weight='balanced',random_state=42),
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

dataset = df.to_dict(orient='index')
feature_list = ['poi'] + df_features.columns.tolist()


best_estimator = None  # will contain final best algorithm
best_score = 0  # will contain final estimated best performance score

# perform actual iteration and testing (GridSearch)
print('\n\n==============\nChoose Bset Model for each Algorithm (F1 Score):\n')
warnings.filterwarnings('ignore') # ignore f1 error warning
for ii, algorithm in enumerate(algorithms):
    # GridSearch also performs CrossValidation to automatically generate training and testing sets of data
    model = GridSearchCV(algorithm, params[ii],scoring = 'f1',cv=5)
    model.fit(df_features, df_labels)

    # Evaluate every model

    best_estimator_ii = model.best_estimator_
    best_score_ii = model.best_score_

    print('------------\nF1 Score:',best_score_ii,'\n')

    test_classifier(best_estimator_ii, dataset, feature_list)

    if best_score_ii > best_score:
        best_estimator = best_estimator_ii
        best_score = best_score_ii


clf =  best_estimator
print('\n\n==============\nThe Highest F1 Score Model:\n',clf)



### Task 6: Dump your classifier, dataset, and features_list

"""
Dump Classifier and Data
"""

dataset = df.to_dict(orient='index')
feature_list = ['poi'] + df_features.columns.tolist()
dump_classifier_and_data(clf, dataset, feature_list )

