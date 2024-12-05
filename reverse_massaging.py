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

# Save features X and outcome Y of the training set in their own dataframes, this is the original unmodified training set
og_x_train = df.loc[indices_train].drop('income', axis=1)
og_y_train = df.loc[indices_train, 'income']
# Save features X and outcome Y of the test set in their own dataframes, there will only be one test set (no variants)
x_test = df.loc[indices_test].drop('income', axis=1)
y_test = df.loc[indices_test, 'income']

#%% Before massaging, calculate chance of having income >50K given each sex and discrimination for training and test sets
# Chance of having income >50K given being male for training set
highincome_male_prob_train = df.loc[indices_train].loc[(df.sex == True) & (df.income == True)].shape[0] / df.loc[indices_train].loc[df.sex == True].shape[0]
print("Chance of having income >50K given being male for training set " + str(highincome_male_prob_train))
# Chance of having income >50K given being female for training set
highincome_female_prob_train = df.loc[indices_train].loc[(df.sex == False) & (df.income == True)].shape[0] / df.loc[indices_train].loc[df.sex == False].shape[0]
print("Chance of having income >50K given being female for training set " + str(highincome_female_prob_train))
# Discrimination for training set
print("Discrimination for training set " + str(highincome_male_prob_train - highincome_female_prob_train))

# Chance of having income >50K given being male for test set
highincome_male_prob_test = df.loc[indices_test].loc[(df.sex == True) & (df.income == True)].shape[0] / df.loc[indices_test].loc[df.sex == True].shape[0]
print("Chance of having income >50K given being male for test set" + str(highincome_male_prob_test))
# Chance of having income >50K given being female for test set
highincome_female_prob_test = df.loc[indices_test].loc[(df.sex == False) & (df.income == True)].shape[0] / df.loc[indices_test].loc[df.sex == False].shape[0]
print("Chance of having income >50K given being female for test set " + str(highincome_female_prob_test))
# Discrimination for test set
discr_test = highincome_male_prob_test - highincome_female_prob_test
print("Discrimination for test set" + str(discr_test))


#%% Reverse massaging to create a more discriminatory training set
# 5-fold CV gridsearch for massaging classifier
clf = GaussianNB()
param_grid = {
    'var_smoothing': np.logspace(0, -7)
}
gridsearch = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=1)
gridsearch.fit(og_x_train, og_y_train)

# Results of grid search
print(gridsearch.best_params_)
print(gridsearch.best_score_)
gnb_og_clf = gridsearch.best_estimator_

# Predict probability scores on training set
scores = gnb_og_clf.predict_proba(og_x_train)[:, 1]

# Create dataframes of candidates for promotion and demotion
# Promotion candidates: males with low income
# Demotion candidates: females with high income
train_scores = df.loc[indices_train, ['sex', 'income']]
train_scores.loc[:, 'score'] = scores

promotion_candidates = train_scores.loc[(train_scores.sex == True) & (train_scores.income == False)].sort_values(
    by='score', ascending=False)
demotion_candidates = train_scores.loc[(train_scores.sex == False) & (train_scores.income == True)].sort_values(
    by='score')

# BAD: demoting females with high income and promoting males with low income will lead to these minority groups
# to become smaller, which can lead to overfitting on the few examples left which will decrease accuracy even more
# (self-fulfilling proficy). This is not the intent of the experiment.
# also error: not enough female candidates with high income to demote to low income to reach double discrimination
# (double discrimination was just chosen arbitrarily)
print('Starting promotions/demotions...')
i = 0
while (df.loc[indices_train].loc[(df.sex == True) & (df.income == True)].shape[0] /
       df.loc[indices_train].loc[df.sex == True].shape[0]) - (
        df.loc[indices_train].loc[(df.sex == False) & (df.income == True)].shape[0] /
        df.loc[indices_train].loc[df.sex == False].shape[0]) < discr_test*2:
    df.loc[promotion_candidates.index[i], 'income'] = True
    df.loc[demotion_candidates.index[i], 'income'] = False
    i += 1
print("Amount of promotions/demotions = " + str(i))

# Statistical parity discrimination of training set after reverse massaging
print("Statistical parity discrimination after massaging: " + str(
    (df.loc[indices_train].loc[(df.sex == True) & (df.income == True)].shape[0] /
     df.loc[indices_train].loc[df.sex == True].shape[0]) - (
            df.loc[indices_train].loc[(df.sex == False) & (df.income == True)].shape[0] /
            df.loc[indices_train].loc[df.sex == False].shape[0])))

# Save features X and outcome Y of the updated discriminatory training data in their own dataframes,
# this will serve as the discriminatory training set
d_x_train = df.loc[indices_train].drop('income', axis=1)
d_y_train = df.loc[indices_train, 'income']


#%% Naive Bayes classifier
# Non-discriminatory model already trained
# Train discriminatory model
gridsearch.fit(d_x_train, d_y_train)

# Results of grid search
print(gridsearch.best_params_)
print(gridsearch.best_score_)
gnb_d_clf = gridsearch.best_estimator_

# Compute accuracies
gnb_og_preds = gnb_og_clf.predict(x_test)
gnb_og_acc = accuracy_score(y_test, gnb_og_preds)
gnb_d_preds = gnb_d_clf.predict(x_test)
gnb_d_acc = accuracy_score(y_test, gnb_d_preds)
"""
gnb_nd_preds = gnb_d_clf.predict(x_test)
gnb_nd_acc = accuracy_score(y_test, gnb_nd_preds)
"""

# Compute statistical parity discriminations
test_preds = pd.DataFrame(x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = gnb_og_preds
gnb_og_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                test_preds.loc[test_preds.sex == True].shape[0]) - (
                         test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                         test_preds.loc[test_preds.sex == False].shape[0])

test_preds = pd.DataFrame(x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = gnb_d_preds
gnb_d_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
               test_preds.loc[test_preds.sex == True].shape[0]) - (
                          test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                          test_preds.loc[test_preds.sex == False].shape[0])
"""
test_preds = pd.DataFrame(x_test.loc[:, 'sex'])
test_preds.loc[:, 'prediction'] = gnb_nd_preds
gnb_nd_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                test_preds.loc[test_preds.sex == True].shape[0]) - (
                         test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                         test_preds.loc[test_preds.sex == False].shape[0])
"""

# Print accuracies & discriminations
print("Accuracy of the original GNB model on the test set: " + str(gnb_og_acc))
print("Discrimination of the original GNB model on the test set: " + str(gnb_og_dscrm))
print("Accuracy of the simulated discriminatory GNB model on the test set: " + str(gnb_d_acc))
print("Discrimination of the simulated discriminatory GNB model on the test set: " + str(gnb_d_dscrm))
"""
print("Accuracy of the simulated non-discriminatory GNB model on the test set: " + str(gnb_nd_acc))
print("Discrimination of the simulated non-discriminatory GNB model on the test set: " + str(gnb_nd_dscrm))
"""
