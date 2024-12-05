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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


#%% Loading data
# fetch dataset https://archive.ics.uci.edu/dataset/20/census+income
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

# Splitting the dataset into train and test sets
percentage_train = 0.8
indices_train = np.array(df.iloc[:int(round(df.shape[0] * percentage_train))].index.values.tolist())
indices_test = np.array(df.iloc[int(round(df.shape[0] * percentage_train)):].index.values.tolist())
#indices_train, indices_test = train_test_split(np.arange(df.shape[0]), test_size=1-percentage_train, random_state=0)
# Not randomly splitting for now because results need to stay constant for every run

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
df = df.assign(income=fnc.generate_dummies(df.income, 'income'))  # <=50k = False, >50K = True

# Save features X and outcome Y of the training set before modification in their own dataframes,
# this will serve as the discriminatory training set
d_x_train = df.loc[indices_train].drop('income', axis=1)
d_y_train = df.loc[indices_train, 'income']
# Save features X and outcome Y of the test set in before modification their own dataframes,
# this will serve as the discriminatory test set
d_x_test = df.loc[indices_test].drop('income', axis=1)
d_y_test = df.loc[indices_test, 'income']


#%% Before eliminating discrimination, calculate chance of having income >50K given each sex and discrimination
# for test set
print("Before eliminating discrimination in test set:")
# Chance of having income >50K given being male for test set
highincome_male_prob_test = df.loc[indices_test].loc[(df.sex == True) & (df.income == True)].shape[0] / df.loc[indices_test].loc[df.sex == True].shape[0]
print("Chance of having income >50K given being male: " + str(highincome_male_prob_test))
# Chance of having income >50K given being female for test set
highincome_female_prob_test = df.loc[indices_test].loc[(df.sex == False) & (df.income == True)].shape[0] / df.loc[indices_test].loc[df.sex == False].shape[0]
print("Chance of having income >50K given being female: " + str(highincome_female_prob_test))
# Discrimination for test set
discr_test = highincome_male_prob_test - highincome_female_prob_test
print("Statistical parity discrimination: " + str(discr_test))


# PS best
# for massaging, stable classifiers like naive bayes have more accuracy and more discrimination, in this case better
# than unstable ones like tree
# probably remove sex attribute for prediction
# weka default parameters were used in kamiran & calders
#%% Remove discrimination in test set through preferential sampling
# Perform 5-fold cross validation grid search to find the optimal hyperparameters
param_grid = {
    'var_smoothing': np.logspace(0, -7)
}
gridsearch = GridSearchCV(GaussianNB(), param_grid, n_jobs=-1, verbose=1)
gridsearch.fit(d_x_test, d_y_test)

# Results of grid search
print(gridsearch.best_params_)
print(gridsearch.best_score_)
test_gnb_clf = gridsearch.best_estimator_

# Predict probability scores on test set
scores = test_gnb_clf.predict_proba(d_x_test)[:, 1]

# Concatenate dataframe with probability scores
#test_scores = pd.concat([df.loc[indices_test, ['sex', 'income']], pd.Series(scores, name='score')], axis=1)
test_scores = df.loc[indices_test, ['sex', 'income']]
test_scores.loc[:, 'score'] = scores

# calculate expected size of each combination of socio-demographic and class
DN_size = int(round((test_scores.loc[(test_scores.sex == False)].shape[0] *
                     test_scores.loc[(test_scores.income == False)].shape[0]) / test_scores.shape[0]))
DP_size = int(round((test_scores.loc[(test_scores.sex == False)].shape[0] *
                     test_scores.loc[(test_scores.income == True)].shape[0]) / test_scores.shape[0]))
FN_size = int(round((test_scores.loc[(test_scores.sex == True)].shape[0] *
                     test_scores.loc[(test_scores.income == False)].shape[0]) / test_scores.shape[0]))
FP_size = int(round((test_scores.loc[(test_scores.sex == True)].shape[0] *
                     test_scores.loc[(test_scores.income == True)].shape[0]) / test_scores.shape[0]))

# Take a subset of each combination of socio-demographic and class with probability scores and sort by score,
# sort the subsets concerning the negative class descending
DN = test_scores.loc[(test_scores.sex == False) & (test_scores.income == False)].sort_values(by='score',
                                                                                             ascending=False)
DP = test_scores.loc[(test_scores.sex == False) & (test_scores.income == True)].sort_values(by='score')
FN = test_scores.loc[(test_scores.sex == True) & (test_scores.income == False)].sort_values(by='score',
                                                                                            ascending=False)
FP = test_scores.loc[(test_scores.sex == True) & (test_scores.income == True)].sort_values(by='score')

# Print expected and actual sizes
print("DN expected size: " + str(DN_size))
print('DN actual size: ' + str(DN.shape[0]))
print("DP expected size: " + str(DP_size))
print('DP actual size: ' + str(DP.shape[0]))
print("FN expected size: " + str(FN_size))
print('FN actual size: ' + str(FN.shape[0]))
print("FP expected size: " + str(FP_size))
print('FP actual size: ' + str(FP.shape[0]))

# As long as the observed size of the combination of the socio-demographic and the class in the test indices
# is smaller than the expected size, duplicate or remove samples
print("Starting DN removals...")
i = 0
while df.loc[indices_test].loc[(df.sex == False) & (df.income == False)].shape[0] > DN_size:
    # Remove record from dataframe
    df.drop(DN.index[i], inplace=True)
    # Remove record's index from test indices
    indices_test = np.delete(indices_test, np.argwhere(indices_test == DN.index[i]))
    i += 1
print(str(i) + ' DN removals')

print("Starting DP duplications...")
i = 0
while df.loc[indices_test].loc[(df.sex == False) & (df.income == True)].shape[0] < DP_size:
    # Duplicate record in dataframe
    row = df.loc[DP.index[i]]
    row.name = np.amax(df.index) + 1
    df = pd.concat([df, pd.DataFrame([row])])
    #df.loc[df.index[-1]+1] = df.loc[DP.index[0]]
    # Add index of duplicated record to test indices
    indices_test = np.append(indices_test, np.amax(df.index))
    i += 1
    # When the amount of records that need to be duplicated exceeds the amount of records with the combination of
    # the socio-demographic and the class, simply reiterate over the subset when the end is reached
    if i >= DP.shape[0]:
        i = 0
print(str(i) + ' DP duplications')

print("Starting FN duplications...")
i = 0
while df.loc[indices_test].loc[(df.sex == True) & (df.income == False)].shape[0] < FN_size:
    # Duplicate record in dataframe
    row = df.loc[FN.index[i]]
    row.name = np.amax(df.index) + 1
    df = pd.concat([df, pd.DataFrame([row])])
    # Add index of duplicated record to test indices
    indices_test = np.append(indices_test, np.amax(df.index))
    i += 1
    # When the amount of records that need to be duplicated exceeds the amount of records with the combination of
    # the socio-demographic and the class, simply reiterate over the subset when the end is reached
    if i >= FN.shape[0]:
        i = 0
print(str(i) + ' FN duplications')

print("Starting FP removals...")
i = 0
while df.loc[indices_test].loc[(df.sex == True) & (df.income == True)].shape[0] > FP_size:
    # Remove record from dataframe
    df.drop(FP.index[i], inplace=True)
    # Remove record's index from test indices
    indices_test = np.delete(indices_test, np.argwhere(indices_test == FP.index[i]))
    i += 1
print(str(i) + ' FP removals')

# Save features X and outcome Y of the updated non-discriminatory test set in their own dataframes
nd_x_test = df.loc[indices_test].drop('income', axis=1)
nd_y_test = df.loc[indices_test, 'income']


#%% After eliminating discrimination, calculate chance of having income >50K given each sex and discrimination
# for test set
print("After eliminating discrimination in test set:")
# Chance of having income >50K given being male for test set
highincome_male_prob_test = df.loc[indices_test].loc[(df.sex == True) & (df.income == True)].shape[0] / df.loc[indices_test].loc[df.sex == True].shape[0]
print("Chance of having income >50K given being male: " + str(highincome_male_prob_test))
# Chance of having income >50K given being female for test set
highincome_female_prob_test = df.loc[indices_test].loc[(df.sex == False) & (df.income == True)].shape[0] / df.loc[indices_test].loc[df.sex == False].shape[0]
print("Chance of having income >50K given being female: " + str(highincome_female_prob_test))
# Discrimination for test set
discr_test = highincome_male_prob_test - highincome_female_prob_test
print("Statistical parity discrimination: " + str(discr_test))


#%% Before eliminating discrimination, calculate chance of having income >50K given each sex and discrimination
# for training set
print("Before eliminating discrimination in training set:")
# Chance of having income >50K given being male for training set
highincome_male_prob_train = df.loc[indices_train].loc[(df.sex == True) & (df.income == True)].shape[0] / df.loc[indices_train].loc[df.sex == True].shape[0]
print("Chance of having income >50K given being male " + str(highincome_male_prob_train))
# Chance of having income >50K given being female for training set
highincome_female_prob_train = df.loc[indices_train].loc[(df.sex == False) & (df.income == True)].shape[0] / df.loc[indices_train].loc[df.sex == False].shape[0]
print("Chance of having income >50K given being female " + str(highincome_female_prob_train))
# Discrimination for training set
print("Statistical parity discrimination " + str(highincome_male_prob_train - highincome_female_prob_train))


#%% Remove discrimination in the training set through massaging
# Perform 5-fold cross validation grid search to find the optimal hyperparameters
# Same gridsearch as earlier
gridsearch.fit(d_x_train, d_y_train)

# Results of grid search
print(gridsearch.best_params_)
print(gridsearch.best_score_)
gnb_d_clf = gridsearch.best_estimator_

# Predict probability scores on training set
scores = gnb_d_clf.predict_proba(d_x_train)[:, 1]

# Create dataframes of candidates for promotion and demotion
# Promotion candidates: females with low income
# Demotion candidates: males with high income
#train_scores = pd.concat([df.loc[indices_train, ['sex', 'income']], pd.Series(scores, name='score')], axis=1)
train_scores = df.loc[indices_train, ['sex', 'income']]
train_scores.loc[:, 'score'] = scores

promotion_candidates = train_scores.loc[(train_scores.sex == False) & (train_scores.income == False)].sort_values(
    by='score', ascending=False)
demotion_candidates = train_scores.loc[(train_scores.sex == True) & (train_scores.income == True)].sort_values(
    by='score')

# As long as the statistical parity discrimination of training set is bigger than 0, keep iterating
print('Starting promotions/demotions...')
i = 0
while (df.loc[indices_train].loc[(df.sex == True) & (df.income == True)].shape[0] /
       df.loc[indices_train].loc[df.sex == True].shape[0]) - (
        df.loc[indices_train].loc[(df.sex == False) & (df.income == True)].shape[0] /
        df.loc[indices_train].loc[df.sex == False].shape[0]) > 0:
    df.loc[promotion_candidates.index[i], 'income'] = True
    df.loc[demotion_candidates.index[i], 'income'] = False
    i += 1
print("Amount of promotions/demotions = " + str(i))

# Save features X and outcome Y of the updated non-discriminatory training data in their own dataframes
nd_x_train = df.loc[indices_train].drop('income', axis=1)
nd_y_train = df.loc[indices_train, 'income']


#%% After eliminating discrimination, calculate chance of having income >50K given each sex and discrimination
# for training set
print("After eliminating discrimination in training set:")
# Chance of having income >50K given being male for training set
highincome_male_prob_train = df.loc[indices_train].loc[(df.sex == True) & (df.income == True)].shape[0] / df.loc[indices_train].loc[df.sex == True].shape[0]
print("Chance of having income >50K given being male " + str(highincome_male_prob_train))
# Chance of having income >50K given being female for training set
highincome_female_prob_train = df.loc[indices_train].loc[(df.sex == False) & (df.income == True)].shape[0] / df.loc[indices_train].loc[df.sex == False].shape[0]
print("Chance of having income >50K given being female " + str(highincome_female_prob_train))
# Discrimination for training set
print("Statistical parity discrimination " + str(highincome_male_prob_train - highincome_female_prob_train))


#%% Naive Bayes classifier
# Discriminatory model already trained
# Train non-discriminatory model (on massaged dataset): same gridsearch as earlier
gridsearch.fit(nd_x_train, nd_y_train)

# Results of grid search
print(gridsearch.best_params_)
print(gridsearch.best_score_)
gnb_nd_clf = gridsearch.best_estimator_

# Compute accuracies
gnb_d_d_preds = gnb_d_clf.predict(d_x_test)
gnb_d_d_acc = accuracy_score(d_y_test, gnb_d_d_preds)
gnb_nd_d_preds = gnb_nd_clf.predict(d_x_test)
gnb_nd_d_acc = accuracy_score(d_y_test, gnb_nd_d_preds)
gnb_d_nd_preds = gnb_d_clf.predict(nd_x_test)
gnb_d_nd_acc = accuracy_score(nd_y_test, gnb_d_nd_preds)
gnb_nd_nd_preds = gnb_nd_clf.predict(nd_x_test)
gnb_nd_nd_acc = accuracy_score(nd_y_test, gnb_nd_nd_preds)

# Compute statistical parity discriminations
test_preds = pd.DataFrame(d_x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = gnb_d_d_preds
gnb_d_d_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                 test_preds.loc[test_preds.sex == True].shape[0]) - (
                         test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                         test_preds.loc[test_preds.sex == False].shape[0])

test_preds = pd.DataFrame(d_x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = gnb_nd_d_preds
gnb_nd_d_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                  test_preds.loc[test_preds.sex == True].shape[0]) - (
                          test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                          test_preds.loc[test_preds.sex == False].shape[0])

test_preds = pd.DataFrame(nd_x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = gnb_d_nd_preds
gnb_d_nd_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                  test_preds.loc[test_preds.sex == True].shape[0]) - (
                         test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                         test_preds.loc[test_preds.sex == False].shape[0])

test_preds = pd.DataFrame(nd_x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = gnb_nd_nd_preds
gnb_nd_nd_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                   test_preds.loc[test_preds.sex == True].shape[0]) - (
                          test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                          test_preds.loc[test_preds.sex == False].shape[0])

# Print accuracies & discriminations
print("Accuracy of the base GNB model on the test set: " + str(gnb_d_d_acc))
print("Discrimination of the base GNB model on the test set: " + str(gnb_d_d_dscrm))
print("Accuracy of the non-discriminatory GNB model on the test set: " + str(gnb_nd_d_acc))
print("Discrimination of the non-discriminatory GNB model on the test set: " + str(gnb_nd_d_dscrm))
print("Accuracy of the base GNB model on the simulated non-discriminatory test set: " + str(gnb_d_nd_acc))
print("Discrimination of the base GNB model on the simulated non-discriminatory test set: " + str(gnb_d_nd_dscrm))
print("Accuracy of the non-discriminating GNB model on the simulated non-discriminatory test set: " +
      str(gnb_nd_nd_acc))
print("Discrimination of the non-discriminating GNB model on the simulated non-discriminatory test set: " +
      str(gnb_nd_nd_dscrm))


#%% Decision tree classifier
# Train model on original discriminatory training set: perform 5-fold cross validation grid search to find the
# optimal hyperparameters
param_grid = {
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [20, 50, 100, 150],
    'min_samples_leaf': [10, 25, 50, 75],
    'max_features': [None, 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
}
gridsearch = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid, n_jobs=-1, verbose=1)
gridsearch.fit(d_x_train, d_y_train)

# Results of grid search
print(gridsearch.best_params_)
print(gridsearch.best_score_)
dt_d_clf = gridsearch.best_estimator_

# Train model on massaged training set: Perform 5-fold cross validation grid search to find the optimal hyperparameters
gridsearch.fit(nd_x_train, nd_y_train)

# Results of grid search
print(gridsearch.best_params_)
print(gridsearch.best_score_)
dt_nd_clf = gridsearch.best_estimator_

# Compute accuracy of both models on discriminatory test set
dt_d_d_preds = dt_d_clf.predict(d_x_test)
dt_d_d_acc = accuracy_score(d_y_test, dt_d_d_preds)
dt_nd_d_preds = dt_nd_clf.predict(d_x_test)
dt_nd_d_acc = accuracy_score(d_y_test, dt_nd_d_preds)

# Compute accuracy of both models on non-discriminatory test set
dt_d_nd_preds = dt_d_clf.predict(nd_x_test)
dt_d_nd_acc = accuracy_score(nd_y_test, dt_d_nd_preds)
dt_nd_nd_preds = dt_nd_clf.predict(nd_x_test)
dt_nd_nd_acc = accuracy_score(nd_y_test, dt_nd_nd_preds)

# Compute statistical parity discrimination of both models on discriminatory test set
test_preds = pd.DataFrame(d_x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = dt_d_d_preds
dt_d_d_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                test_preds.loc[test_preds.sex == True].shape[0]) - (
                        test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                        test_preds.loc[test_preds.sex == False].shape[0])

test_preds = pd.DataFrame(d_x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = dt_nd_d_preds
dt_nd_d_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                 test_preds.loc[test_preds.sex == True].shape[0]) - (
                         test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                         test_preds.loc[test_preds.sex == False].shape[0])

# Compute statistical parity discrimination of both models on non-discriminatory test set
test_preds = pd.DataFrame(nd_x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = dt_d_nd_preds
dt_d_nd_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                 test_preds.loc[test_preds.sex == True].shape[0]) - (
                        test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                        test_preds.loc[test_preds.sex == False].shape[0])

test_preds = pd.DataFrame(nd_x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = dt_nd_nd_preds
dt_nd_nd_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                  test_preds.loc[test_preds.sex == True].shape[0]) - (
                         test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                         test_preds.loc[test_preds.sex == False].shape[0])

# Print accuracies & discriminations
print("Accuracy of the base model on the test set: " + str(dt_d_d_acc))
print("Discrimination of the base model on the test set: " + str(dt_d_d_dscrm))
print("Accuracy of the non-discriminatory model on the test set: " + str(dt_nd_d_acc))
print("Discrimination of the non-discriminatory model on the test set: " + str(dt_nd_d_dscrm))
print("Accuracy of the base model on the simulated non-discriminatory test set: " + str(dt_d_nd_acc))
print("Discrimination of the base model on the simulated non-discriminatory test set: " + str(dt_d_nd_dscrm))
print("Accuracy of the non-discriminating model on the simulated non-discriminatory test set: " + str(dt_nd_nd_acc))
print(
    "Discrimination of the non-discriminatory model on the simulated non-discriminatory test set: " + str(
        dt_nd_nd_dscrm))


#%% K-nearest neighbors classifier
# Train model on original discriminatory training set: Perform 5-fold CV grid search to find the optimal
# hyperparameters
param_grid = {
    'n_neighbors': [15, 20, 30, 40, 50],
    'weights': ['uniform', 'distance'],
    'leaf_size': [5, 10, 15]
}
gridsearch = GridSearchCV(KNeighborsClassifier(n_jobs=-1), param_grid, n_jobs=-1, verbose=1)
gridsearch.fit(d_x_train, d_y_train)

# Results of grid search
print(gridsearch.best_params_)
print(gridsearch.best_score_)
knn_d_clf = gridsearch.best_estimator_

# Train model on massaged training set: Perform 5-fold CV grid search to find the optimal hyperparameters
gridsearch.fit(nd_x_train, nd_y_train)

# Results of grid search
print(gridsearch.best_params_)
print(gridsearch.best_score_)
knn_nd_clf = gridsearch.best_estimator_

# Compute accuracies
knn_d_d_preds = knn_d_clf.predict(d_x_test)
knn_d_d_acc = accuracy_score(d_y_test, knn_d_d_preds)
knn_nd_d_preds = knn_nd_clf.predict(d_x_test)
knn_nd_d_acc = accuracy_score(d_y_test, knn_nd_d_preds)
knn_d_nd_preds = knn_d_clf.predict(nd_x_test)
knn_d_nd_acc = accuracy_score(nd_y_test, knn_d_nd_preds)
knn_nd_nd_preds = knn_nd_clf.predict(nd_x_test)
knn_nd_nd_acc = accuracy_score(nd_y_test, knn_nd_nd_preds)

# Compute statistical parity discriminations
test_preds = pd.DataFrame(d_x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = knn_d_d_preds
knn_d_d_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                 test_preds.loc[test_preds.sex == True].shape[0]) - (
                         test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                         test_preds.loc[test_preds.sex == False].shape[0])

test_preds = pd.DataFrame(d_x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = knn_nd_d_preds
knn_nd_d_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                  test_preds.loc[test_preds.sex == True].shape[0]) - (
                          test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                          test_preds.loc[test_preds.sex == False].shape[0])

test_preds = pd.DataFrame(nd_x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = knn_d_nd_preds
knn_d_nd_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                  test_preds.loc[test_preds.sex == True].shape[0]) - (
                         test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                         test_preds.loc[test_preds.sex == False].shape[0])

test_preds = pd.DataFrame(nd_x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = knn_nd_nd_preds
knn_nd_nd_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                   test_preds.loc[test_preds.sex == True].shape[0]) - (
                          test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                          test_preds.loc[test_preds.sex == False].shape[0])

# Print accuracies & discriminations
print("Accuracy of the base KNN model on the test set: " + str(knn_d_d_acc))
print("Discrimination of the base KNN model on the test set: " + str(knn_d_d_dscrm))
print("Accuracy of the non-discriminatory KNN model on the test set: " + str(knn_nd_d_acc))
print("Discrimination of the non-discriminatory KNN model on the test set: " + str(knn_nd_d_dscrm))
print("Accuracy of the base KNN model on the simulated non-discriminatory test set: " + str(knn_d_nd_acc))
print("Discrimination of the base model on the simulated non-discriminatory test set: " + str(knn_d_nd_dscrm))
print("Accuracy of the non-discriminatory KNN model on the simulated non-discriminatory test set: " +
      str(knn_nd_nd_acc))
print("Discrimination of the non-discriminatory KNN model on the simulated non-discriminatory test set: " +
      str(knn_nd_nd_dscrm))

#%% Visualizing results
# Set title and label sizes
plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.titlesize'] = 12

# Figure accuracies & discriminations of models on discriminatory test set
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Subplot accuracies
acc_d_d = [dt_d_d_acc, knn_d_d_acc, gnb_d_d_acc]
acc_nd_nd = [dt_nd_d_acc, knn_nd_d_acc, gnb_nd_d_acc]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminatory model': acc_d_d,
               'Non-discriminatory model': acc_nd_nd}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classification algorithm',
           ylabel='Accuracy (%)', rot=0, ax=axes[0]))

# Subplot discriminations
dscrm_d_d = [dt_d_d_dscrm, knn_d_d_dscrm, gnb_d_d_dscrm]
dscrm_nd_d = [dt_nd_d_dscrm, knn_nd_d_dscrm, gnb_nd_d_dscrm]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminatory model': dscrm_d_d,
               'Non-discriminatory model': dscrm_nd_d}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classification algorithm',
           ylabel='Discrimination (%)', rot=0, ax=axes[1]))

fig.suptitle('Accuracy and discrimination of the discriminatory and\n'
             'non-discriminatory model for each algorithm on the discriminatory test set')
axes[0].tick_params(axis='y', direction='in', right=True)
axes[1].tick_params(axis='y', direction='in', right=True)
plt.tight_layout()
plt.show()

# Figure accuracies & discriminations of models on non-discriminatory test set
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Subplot accuracies
acc_d_nd = [dt_d_nd_acc, knn_d_nd_acc, gnb_d_nd_acc]
acc_nd_nd = [dt_nd_nd_acc, knn_nd_nd_acc, gnb_nd_nd_acc]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminatory model': acc_d_nd,
               'Non-discriminatory model': acc_nd_nd}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classification algorithm',
           ylabel='Accuracy (%)', rot=0, ax=axes[0]))

# Subplot discriminations
dscrm_d_nd = [dt_d_nd_dscrm, knn_d_nd_dscrm, gnb_d_nd_dscrm]
dscrm_nd_nd = [dt_nd_nd_dscrm, knn_nd_nd_dscrm, gnb_nd_nd_dscrm]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminatory model': dscrm_d_nd,
               'Non-discriminatory model': dscrm_nd_nd}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classification algorithm',
           ylabel='Discrimination (%)', rot=0, ax=axes[1]))

fig.suptitle('Accuracy and discrimination of the discriminatory and non-discriminatory\n'
             'model for each algorithm on the non-discriminatory test set')
axes[0].tick_params(axis='y', direction='in', right=True)
axes[1].tick_params(axis='y', direction='in', right=True)
plt.tight_layout()
plt.show()

"""
#%% Visualizing results Nederlands (zelfde grafieken)
# Figure accuracies & discriminations of models on discriminatory test set
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Subplot accuracies
acc_d_d = [dt_d_d_acc, knn_d_d_acc, gnb_d_d_acc]
acc_nd_d = [dt_nd_d_acc, knn_nd_d_acc, gnb_nd_d_acc]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminerend model': acc_d_d,
               'Non-discriminerend model': acc_nd_d}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classificatiealgoritme',
           ylabel='Accuraatheid (%)', rot=0, ax=axes[0]))

# Subplot discriminations
dscrm_d_d = [dt_d_d_dscrm, knn_d_d_dscrm, gnb_d_d_dscrm]
dscrm_nd_d = [dt_nd_d_dscrm, knn_nd_d_dscrm, gnb_nd_d_dscrm]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminerend model': dscrm_d_d,
               'Non-discriminerend model': dscrm_nd_d}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classificatiealgoritme',
           ylabel='Mate van discriminatie (%)', rot=0, ax=axes[1]))

fig.suptitle('Accuraatheid en discriminatie van het discriminerend en\n'
             'non-discriminerend model voor elk algoritme op de discriminerende test set')
axes[0].tick_params(axis='y', direction='in', right=True)
axes[1].tick_params(axis='y', direction='in', right=True)
plt.tight_layout()
plt.show()

# Figure accuracies & discriminations of models on non-discriminatory test set
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Subplot accuracies
acc_d_nd = [dt_d_nd_acc, knn_d_nd_acc, gnb_d_nd_acc]
acc_nd_nd = [dt_nd_nd_acc, knn_nd_nd_acc, gnb_nd_nd_acc]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminerend model': acc_d_nd,
               'Non-discriminerend model': acc_nd_nd}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classificatiealgoritme',
           ylabel='Accuraatheid (%)', rot=0, ax=axes[0]))

# Subplot discriminations
dscrm_d_nd = [dt_d_nd_dscrm, knn_d_nd_dscrm, gnb_d_nd_dscrm]
dscrm_nd_nd = [dt_nd_nd_dscrm, knn_nd_nd_dscrm, gnb_nd_nd_dscrm]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminerend model': dscrm_d_nd,
               'Non-discriminerend model': dscrm_nd_nd}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classificatiealgoritme',
           ylabel='Mate van discriminatie (%)', rot=0, ax=axes[1]))

fig.suptitle('Accuraatheid en discriminatie van het discriminerend en non-discriminerend\n'
             'model voor elk algoritme op de non-discriminerende test set')
axes[0].tick_params(axis='y', direction='in', right=True)
axes[1].tick_params(axis='y', direction='in', right=True)
plt.tight_layout()
plt.show()
"""


#%% Math theorem test
# Probability of high income given male in train set - same probability in test set * proportion of men in train -
# prob of high income given female in train set - same prob in test set * proportion of female in train
# then same for non-discriminatory train set vs test set (already non-discriminatory), this number should be lower (?)
# haakjes rond kans hoog inkomen man train - zelfde kans test?
train = pd.concat([d_x_train, d_y_train], axis=1)
test = pd.concat([nd_x_test, nd_y_test], axis=1)
nd_train = pd.concat([nd_x_train, nd_y_train], axis=1)

a = (train.loc[(train.sex == True) & (train.income == True)].shape[0] / train.loc[train.sex == True].shape[0]) - (
        test.loc[(test.sex == True) & (test.income == True)].shape[0] / test.loc[test.sex == True].shape[0]) * (
        train.loc[train.sex == True].shape[0] / train.shape[0]) - (train.loc[(train.sex == False) & (train.income == True)].shape[0] / train.loc[train.sex == False].shape[0]) - (
        test.loc[(test.sex == False) & (test.income == True)].shape[0] / test.loc[test.sex == False].shape[0]) * (
        train.loc[train.sex == False].shape[0] / train.shape[0])
b = (nd_train.loc[(nd_train.sex == True) & (nd_train.income == True)].shape[0] / nd_train.loc[nd_train.sex == True].shape[0]) - (
        test.loc[(test.sex == True) & (test.income == True)].shape[0] / test.loc[test.sex == True].shape[0]) * (
        nd_train.loc[nd_train.sex == True].shape[0] / nd_train.shape[0]) - (nd_train.loc[(nd_train.sex == False) & (nd_train.income == True)].shape[0] / nd_train.loc[nd_train.sex == False].shape[0]) - (
        test.loc[(test.sex == False) & (test.income == True)].shape[0] / test.loc[test.sex == False].shape[0]) * (
        nd_train.loc[nd_train.sex == False].shape[0] / nd_train.shape[0])
print(a)
print(b)
