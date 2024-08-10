#%%
#pip install ucimlrepo

#%% Imports
import sys
from ucimlrepo import fetch_ucirepo
import numpy as np
import functions as fnc
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

np.set_printoptions(threshold=sys.maxsize)

#%% Loading data
# fetch dataset
census_income = fetch_ucirepo(id=20)
df = census_income.data.original

# metadata
print(census_income.metadata)

# variable information
print(census_income.variables)

#%% Data analysis

# Amount of observations
print(df.shape[0])  # 48842

# Check frequency distributions & invalid values
print(df.age.value_counts().sort_index())
print(df.workclass.value_counts().sort_values())
print(df.fnlwgt.value_counts().sort_values())
print(df.education.value_counts().sort_values())
print(df['education-num'].value_counts().sort_index())
print(df['marital-status'].value_counts().sort_values())
print(df.occupation.value_counts().sort_values())
print(df.relationship.value_counts().sort_values())
print(df.race.value_counts().sort_values())
print(df.sex.value_counts().sort_values())
print(df['capital-gain'].value_counts().sort_index())
print(df['capital-loss'].value_counts().sort_index())
print(df['hours-per-week'].value_counts().sort_index())
print(df['native-country'].value_counts().sort_values())
print(df.income.value_counts().sort_values())

# The income variable has 4 different variables, but Kamiran & Calders (2011) treated this as a binary classification
# problem. Two of the values also seemingly make no sense and seem to be a submission error. These need to be fixed.

# Check missing values
print(len(np.argwhere(df.age.isna().tolist())))  # 0 missings
print(len(np.argwhere(df.workclass.isna().tolist())))  # 963 missings
print(len(np.argwhere(df.fnlwgt.isna().tolist())))
print(len(np.argwhere(df.education.isna().tolist())))  # 0 missings
print(len(np.argwhere(df['education-num'].isna().tolist())))  # 0 missings
print(len(np.argwhere(df['marital-status'].isna().tolist())))  # 0 missings
print(len(np.argwhere(df.occupation.isna().tolist())))  # 966 missings
print(len(np.argwhere(df.relationship.isna().tolist())))  # 0 missings
print(len(np.argwhere(df.race.isna().tolist())))  # 0 missings
print(len(np.argwhere(df.sex.isna().tolist())))  # 0 missings
print(len(np.argwhere(df['capital-gain'].isna().tolist())))  # 0 missings
print(len(np.argwhere(df['capital-loss'].isna().tolist())))  # 0 missings
print(len(np.argwhere(df['hours-per-week'].isna().tolist())))  # 0 missings
print(len(np.argwhere(df['native-country'].isna().tolist())))  # 274 missings
print(len(np.argwhere(df.income.isna().tolist())))  # 0 missings

# workclass, occupation and native-country are categorical variables and have a fairly low amount of missings, so
# remove these observations with missings.

#%% Data preprocessing

# Dropping detrimental data
# Kamiran & Calders (2011) suggest removing the variable fnlwgt
df.drop('fnlwgt', axis=1, inplace=True)
df.drop(df.loc[df.workclass.isna()].index.values.tolist(), inplace=True)
df.drop(df.loc[df.occupation.isna()].index.values.tolist(), inplace=True)
df.drop(df.loc[df['native-country'].isna()].index.values.tolist(), inplace=True)

# Reset dataframe index after dropping records
df.reset_index(drop=True, inplace=True)

# Correcting the values of the income variable
df.loc[df.income == '>50K.', 'income'] = '>50K'
df.loc[df.income == '<=50K.', 'income'] = '<=50K'

# Sort the dataset so that when splitting the test split will have less discrimination
df.sort_values(by='age', ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

# Splitting the dataset into train and test sets
percentage_train = 0.8
indices_train = np.array(df.iloc[:int(round(df.shape[0] * percentage_train))].index.values.tolist())
indices_test = np.array(df.iloc[int(round(df.shape[0] * percentage_train)):].index.values.tolist())
#indices_train, indices_test = train_test_split(np.arange(df.shape[0]), test_size=1-percentage_train, random_state=0)

# Calculate chance of having income >50K for both sexes and discrimination for training and test sets
print(df.loc[indices_train].loc[(df.sex == 'Male') & (df.income == '>50K')].shape[0] /
      df.loc[indices_train].loc[df.sex == 'Male'].shape[0])
print(df.loc[indices_train].loc[(df.sex == 'Female') & (df.income == '>50K')].shape[0] /
      df.loc[indices_train].loc[df.sex == 'Female'].shape[0])
print((df.loc[indices_train].loc[(df.sex == 'Male') & (df.income == '>50K')].shape[0] /
      df.loc[indices_train].loc[df.sex == 'Male'].shape[0]) - (df.loc[indices_train].loc[(df.sex == 'Female') & (df.income == '>50K')].shape[0] /
      df.loc[indices_train].loc[df.sex == 'Female'].shape[0]))

print(df.loc[indices_test].loc[(df.sex == 'Male') & (df.income == '>50K')].shape[0] /
      df.loc[indices_test].loc[df.sex == 'Male'].shape[0])
print(df.loc[indices_test].loc[(df.sex == 'Female') & (df.income == '>50K')].shape[0] /
      df.loc[indices_test].loc[df.sex == 'Female'].shape[0])
print((df.loc[indices_test].loc[(df.sex == 'Male') & (df.income == '>50K')].shape[0] /
      df.loc[indices_test].loc[df.sex == 'Male'].shape[0]) - (df.loc[indices_test].loc[(df.sex == 'Female') & (df.income == '>50K')].shape[0] /
      df.loc[indices_test].loc[df.sex == 'Female'].shape[0]))

# Normalizing discrete/continuous variables
df = df.assign(age=fnc.normalize(df.age, indices_train))
df = df.assign(education_num=fnc.normalize(df['education-num'], indices_train))
df = df.assign(capital_gain=fnc.normalize(df['capital-gain'], indices_train))
df = df.assign(capital_loss=fnc.normalize(df['capital-loss'], indices_train))
df = df.assign(hours_per_week=fnc.normalize(df['hours-per-week'], indices_train))
df.drop('education-num', axis=1, inplace=True)
df.drop('capital-gain', axis=1, inplace=True)
df.drop('capital-loss', axis=1, inplace=True)
df.drop('hours-per-week', axis=1, inplace=True)

# Converting categorical variables to dummy variables for k-1 unique values
df = pd.concat([df, fnc.generate_dummies(df.workclass, 'workclass')], axis=1)
df.drop('workclass', axis=1, inplace=True)
df = pd.concat([df, fnc.generate_dummies(df.education, 'education')], axis=1)
df.drop('education', axis=1, inplace=True)
df = pd.concat([df, fnc.generate_dummies(df['marital-status'], 'marital-status')], axis=1)
df.drop('marital-status', axis=1, inplace=True)
df = pd.concat([df, fnc.generate_dummies(df.occupation, 'occupation')], axis=1)
df.drop('occupation', axis=1, inplace=True)
df = pd.concat([df, fnc.generate_dummies(df.relationship, 'relationship')], axis=1)
df.drop('relationship', axis=1, inplace=True)
df = pd.concat([df, fnc.generate_dummies(df.race, 'race')], axis=1)
df.drop('race', axis=1, inplace=True)
df = df.assign(sex=fnc.generate_dummies(df.sex, 'sex'))  # female = False, male = True
df = pd.concat([df, fnc.generate_dummies(df['native-country'], 'native-country')], axis=1)
df.drop('native-country', axis=1, inplace=True)
df = df.assign(income=fnc.generate_dummies(df.income, 'income')) # <=50k = False, >50K = True

#%% Remove discrimination in the training set through massaging
# Assign train and test instances of features X and outcome Y
x_train = df.loc[indices_train].drop('income', axis=1)
y_train = df.loc[indices_train, 'income']

x_test = df.loc[indices_test].drop('income', axis=1)
y_test = df.loc[indices_test, 'income']

# Perform 5-fold cross validation grid search to find the optimal hyperparameters
clf = DecisionTreeClassifier(random_state=17)
param_grid = {
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [20, 50, 100, 150],
    'min_samples_leaf': [10, 25, 50, 75],
    'max_features': [None, 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
}
gridsearch = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=1)
gridsearch.fit(x_train, y_train)

# Results of grid search
print(gridsearch.best_params_)
print(gridsearch.best_score_)
clf = gridsearch.best_estimator_

# Compute accuracy on test set
accuracy = accuracy_score(y_test, clf.predict(x_test))
print("Accuracy of the base model on the test set: " + str(accuracy))

# Predict probability scores on training set
scores = clf.predict_proba(x_train)[:, 1]

# create dataframes of candidates for promotion and demotion
train_scores = pd.concat([df.loc[indices_train, ['sex', 'income']], pd.Series(scores, name='score')], axis=1)

promotion_candidates = train_scores.loc[(train_scores.sex == False) & (train_scores.income == False)].sort_values(by='score', ascending=False)
demotion_candidates = train_scores.loc[(train_scores.sex == True) & (train_scores.income == True)].sort_values(by='score')

# Statistical parity discrimination of training set
print((df.loc[indices_train].loc[(df.sex == True) & (df.income == True)].shape[0] /
    df.loc[indices_train].loc[df.sex == True].shape[0]) - (df.loc[indices_train].loc[(df.sex == False) & (df.income == True)].shape[0] /
       df.loc[indices_train].loc[df.sex == False].shape[0]))

# Statistical parity discrimination of test set
print((df.loc[indices_test].loc[(df.sex == True) & (df.income == True)].shape[0] /
    df.loc[indices_test].loc[df.sex == True].shape[0]) - (df.loc[indices_test].loc[(df.sex == False) & (df.income == True)].shape[0] /
      df.loc[indices_test].loc[df.sex == False].shape[0]))

# As long as the statistical parity discrimination of training set is bigger than that of the test set, keep iterating
print('Starting promotions/demotions...')
i = 0
while (df.loc[indices_train].loc[(df.sex == True) & (df.income == True)].shape[0] /
    df.loc[indices_train].loc[df.sex == True].shape[0]) - (df.loc[indices_train].loc[(df.sex == False) & (df.income == True)].shape[0] /
       df.loc[indices_train].loc[df.sex == False].shape[0]) > (df.loc[indices_test].loc[(df.sex == True) & (df.income == True)].shape[0] /
    df.loc[indices_test].loc[df.sex == True].shape[0]) - (df.loc[indices_test].loc[(df.sex == False) & (df.income == True)].shape[0] /
      df.loc[indices_test].loc[df.sex == False].shape[0]):
    df.loc[promotion_candidates.index[i], 'income'] = True
    df.loc[demotion_candidates.index[i], 'income'] = False
    i += 1
print("Amount of promotions/demotions = " + str(i))

# Reinitialize training set with updated data
x_train = df.loc[indices_train].drop('income', axis=1)
y_train = df.loc[indices_train, 'income']

# Perform grid search to find the optimal hyperparameters
nd_clf = DecisionTreeClassifier(random_state=17)
param_grid = {
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [20, 50, 100, 150],
    'min_samples_leaf': [10, 25, 50, 75],
    'max_features': [None, 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
}
gridsearch = GridSearchCV(nd_clf, param_grid, n_jobs=-1, verbose=1)
gridsearch.fit(x_train, y_train)

# Results of grid search
print(gridsearch.best_params_)
print(gridsearch.best_score_)
nd_clf = gridsearch.best_estimator_

# Compute accuracy on test set
accuracy2 = accuracy_score(y_test, nd_clf.predict(x_test))
print("Accuracy of the non-discriminatory model on the test set: " + str(accuracy2))

# experiment doesn't really work for this dataset because while massaging the dataset gets the probability of the
# positive class given being female closer to that of the test set, it gets the probability of the positive class
# given being male further away from that of the test set, so the accuracy decreases. This is the case both when
# sorting descendingly and when sorting ascendingly and massaging the test set instead of the training set.
