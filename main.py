#%%
#pip install ucimlrepo

#%% Imports
from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

#%% Loading data
# fetch dataset
census_income = fetch_ucirepo(id=20)
df = census_income.data.original

# metadata
print(census_income.metadata)

# variable information
print(census_income.variables)


#%% Data analysis

# Checking frequency distributions
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

#%% Data preprocessing

df = df.drop('fnlwgt', axis=1)
df.loc[df.income == '>50K.', 'income'] = '>50K'
df.loc[df.income == '<=50K.', 'income'] = '<=50K'


#%% First experiment:

percentage_train = 0.8
indices_train = np.array(df.iloc[:int(round(df.shape[0]*percentage_train))].index.values.tolist())
indices_test = np.array(df.iloc[int(round(df.shape[0]*percentage_train)):].index.values.tolist())
#indices_train, indices_test = train_test_split(np.arange(df.shape[0]), test_size=1-percentage_train, random_state=0)

print(df.loc[indices_train].loc[(df.sex == 'Female') & (df.income == '>50K')].shape[0]/df.loc[indices_train].loc[df.sex == 'Female'].shape[0])
print(df.loc[indices_train].loc[(df.sex == 'Male') & (df.income == '>50K')].shape[0]/df.loc[indices_train].loc[df.sex == 'Male'].shape[0])

print(df.loc[indices_test].loc[(df.sex == 'Female') & (df.income == '>50K')].shape[0]/df.loc[indices_test].loc[df.sex == 'Female'].shape[0])
print(df.loc[indices_test].loc[(df.sex == 'Male') & (df.income == '>50K')].shape[0]/df.loc[indices_test].loc[df.sex == 'Male'].shape[0])

x_train = df.loc[indices_train].drop('income', axis=1)
y_train = df.loc[indices_train, 'income']

x_test = df.loc[indices_test].drop('income', axis=1)
y_test = df.loc[indices_test, 'income']

clf = DecisionTreeClassifier()
param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}
gridsearch = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2)
gridsearch.fit(x_train, y_train)

print(gridsearch.best_params_)

print('lol')
