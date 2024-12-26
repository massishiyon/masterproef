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
import techniques as tch

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

# PS best
# for massaging, stable classifiers like naive bayes have more accuracy and more discrimination, in this case better
# than unstable ones like tree
# probably remove sex attribute for prediction
# weka default parameters were used in kamiran & calders
#%% Remove discrimination in test set to simulate non-discriminatory test set
print("Eliminating discrimination in test set")
nd_ps_gnb_x_test, nd_ps_gnb_y_test = tch.preferential_sampling(df, indices_test, d_x_test, d_y_test, "GNB")[0:2]
nd_mas_gnb_x_test, nd_mas_gnb_y_test = tch.massaging(df, indices_test, d_x_test, d_y_test, "GNB")[0:2]

#%% Remove discrimination in the training set to make model non-discriminatory
print("Eliminating discrimination in training set")
# The first time a classifier of a certain algorithm is learned on the training set, it needs to be returned for
# later use
nd_ps_gnb_x_train, nd_ps_gnb_y_train = tch.preferential_sampling(df, indices_train, d_x_train, d_y_train, "GNB")[0:2]
nd_mas_gnb_x_train, nd_mas_gnb_y_train, gnb_d_clf = tch.massaging(df, indices_train, d_x_train, d_y_train, "GNB")

#%% Naive Bayes classifier
(gnb_d_d_acc, gnb_nd_d_acc, gnb_d_nd_ps_gnb_acc, gnb_nd_mas_gnb_nd_ps_gnb_acc, gnb_nd_ps_gnb_nd_ps_gnb_acc,
 gnb_d_nd_mas_gnb_acc, gnb_nd_mas_gnb_nd_mas_gnb_acc, gnb_nd_ps_gnb_nd_mas_gnb_acc, gnb_d_d_dscrm, gnb_nd_d_dscrm,
 gnb_d_nd_ps_gnb_dscrm, gnb_nd_mas_gnb_nd_ps_gnb_dscrm, gnb_nd_ps_gnb_nd_ps_gnb_dscrm, gnb_d_nd_mas_gnb_dscrm,
 gnb_nd_mas_gnb_nd_mas_gnb_dscrm, gnb_nd_ps_gnb_nd_mas_gnb_dscrm) = (
    fnc.train_and_test_clf(d_x_train, d_y_train, d_x_test, d_y_test, nd_mas_gnb_x_train, nd_mas_gnb_y_train,
                           nd_mas_gnb_x_test, nd_mas_gnb_y_test, nd_ps_gnb_x_train, nd_ps_gnb_y_train, nd_ps_gnb_x_test,
                           nd_ps_gnb_y_test, "GNB", gnb_d_clf))

#%% Decision tree classifier
(dt_d_d_acc, dt_nd_d_acc, dt_d_nd_ps_gnb_acc, dt_nd_mas_gnb_nd_ps_gnb_acc, dt_nd_ps_gnb_nd_ps_gnb_acc,
 dt_d_nd_mas_gnb_acc, dt_nd_mas_gnb_nd_mas_gnb_acc, dt_nd_ps_gnb_nd_mas_gnb_acc, dt_d_d_dscrm, dt_nd_d_dscrm,
 dt_d_nd_ps_gnb_dscrm, dt_nd_mas_gnb_nd_ps_gnb_dscrm, dt_nd_ps_gnb_nd_ps_gnb_dscrm, dt_d_nd_mas_gnb_dscrm,
 dt_nd_mas_gnb_nd_mas_gnb_dscrm, dt_nd_ps_gnb_nd_mas_gnb_dscrm) = (
    fnc.train_and_test_clf(d_x_train, d_y_train, d_x_test, d_y_test, nd_mas_gnb_x_train, nd_mas_gnb_y_train,
                           nd_mas_gnb_x_test, nd_mas_gnb_y_test, nd_ps_gnb_x_train, nd_ps_gnb_y_train, nd_ps_gnb_x_test,
                           nd_ps_gnb_y_test, "DT"))

#%% K-nearest neighbors classifier
(knn_d_d_acc, knn_nd_d_acc, knn_d_nd_ps_gnb_acc, knn_nd_mas_gnb_nd_ps_gnb_acc, knn_nd_ps_gnb_nd_ps_gnb_acc,
 knn_d_nd_mas_gnb_acc, knn_nd_mas_gnb_nd_mas_gnb_acc, knn_nd_ps_gnb_nd_mas_gnb_acc, knn_d_d_dscrm, knn_nd_d_dscrm,
 knn_d_nd_ps_gnb_dscrm, knn_nd_mas_gnb_nd_ps_gnb_dscrm, knn_nd_ps_gnb_nd_ps_gnb_dscrm, knn_d_nd_mas_gnb_dscrm,
 knn_nd_mas_gnb_nd_mas_gnb_dscrm, knn_nd_ps_gnb_nd_mas_gnb_dscrm) = (
    fnc.train_and_test_clf(d_x_train, d_y_train, d_x_test, d_y_test, nd_mas_gnb_x_train, nd_mas_gnb_y_train,
                           nd_mas_gnb_x_test, nd_mas_gnb_y_test, nd_ps_gnb_x_train, nd_ps_gnb_y_train, nd_ps_gnb_x_test,
                           nd_ps_gnb_y_test, "KNN"))

# TODO: remove sex from dataset after training model (in train and test function)

#%% Visualizing results
# Set title and label sizes
plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.titlesize'] = 12

# Visualize comparison of accuracies & discriminations of the discriminatory and a non-discriminatory model on
# discriminatory test set
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#   Subplot accuracies
acc_d_d = [dt_d_d_acc, knn_d_d_acc, gnb_d_d_acc]
acc_nd_d = [dt_nd_d_acc, knn_nd_d_acc, gnb_nd_d_acc]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminatory model': acc_d_d,
               'Non-discriminatory model': acc_nd_d}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classification algorithm',
           ylabel='Accuracy (%)', rot=0, ax=axes[0]))
#   Subplot discriminations
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

# For each non-discriminatory model and each non-discriminatory test set, visualize comparisons of
# accuracies & discriminations of the discriminatory model vs the non-discriminatory model on the non-discriminatory
# test set
#   PS (GNB ranker) ND test set
#       Discriminatory model vs massaged (GNB ranker) ND model
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#           Subplot accuracies
acc_d_nd_ps_gnb = [dt_d_nd_ps_gnb_acc, knn_d_nd_ps_gnb_acc, gnb_d_nd_ps_gnb_acc]
acc_nd_mas_gnb_nd_ps_gnb = [dt_nd_mas_gnb_nd_ps_gnb_acc, knn_nd_mas_gnb_nd_ps_gnb_acc, gnb_nd_mas_gnb_nd_ps_gnb_acc]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminatory model': acc_d_nd_ps_gnb,
               'Non-discriminatory model': acc_nd_mas_gnb_nd_ps_gnb}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classification algorithm',
           ylabel='Accuracy (%)', rot=0, ax=axes[0]))
#           Subplot discriminations
dscrm_d_nd_ps_gnb = [dt_d_nd_ps_gnb_dscrm, knn_d_nd_ps_gnb_dscrm, gnb_d_nd_ps_gnb_dscrm]
dscrm_nd_mas_gnb_nd_ps_gnb = [dt_nd_mas_gnb_nd_ps_gnb_dscrm, knn_nd_mas_gnb_nd_ps_gnb_dscrm,
                              gnb_nd_mas_gnb_nd_ps_gnb_dscrm]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminatory model': dscrm_d_nd_ps_gnb,
               'Non-discriminatory model': dscrm_nd_mas_gnb_nd_ps_gnb}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classification algorithm',
           ylabel='Discrimination (%)', rot=0, ax=axes[1]))

fig.suptitle('Accuracy and discrimination of the discriminatory and massaged (GNB ranker) non-discriminatory\n'
             'model for each algorithm on the PS (GNB ranker) non-discriminatory test set')
axes[0].tick_params(axis='y', direction='in', right=True)
axes[1].tick_params(axis='y', direction='in', right=True)
plt.tight_layout()
plt.show()
#       Discriminatory model vs PS (GNB ranker) ND model
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#           Subplot accuracies
acc_d_nd_ps_gnb = [dt_d_nd_ps_gnb_acc, knn_d_nd_ps_gnb_acc, gnb_d_nd_ps_gnb_acc]
acc_nd_ps_gnb_nd_ps_gnb = [dt_nd_ps_gnb_nd_ps_gnb_acc, knn_nd_ps_gnb_nd_ps_gnb_acc, gnb_nd_ps_gnb_nd_ps_gnb_acc]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminatory model': acc_d_nd_ps_gnb,
               'Non-discriminatory model': acc_nd_ps_gnb_nd_ps_gnb}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classification algorithm',
           ylabel='Accuracy (%)', rot=0, ax=axes[0]))
#           Subplot discriminations
dscrm_d_nd_ps_gnb = [dt_d_nd_ps_gnb_dscrm, knn_d_nd_ps_gnb_dscrm, gnb_d_nd_ps_gnb_dscrm]
dscrm_nd_ps_gnb_nd_ps_gnb = [dt_nd_ps_gnb_nd_ps_gnb_dscrm, knn_nd_ps_gnb_nd_ps_gnb_dscrm, gnb_nd_ps_gnb_nd_ps_gnb_dscrm]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminatory model': dscrm_d_nd_ps_gnb,
               'Non-discriminatory model': dscrm_nd_ps_gnb_nd_ps_gnb}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classification algorithm',
           ylabel='Discrimination (%)', rot=0, ax=axes[1]))

fig.suptitle('Accuracy and discrimination of the discriminatory and PS (GNB ranker) non-discriminatory\n'
             'model for each algorithm on the PS (GNB ranker) non-discriminatory test set')
axes[0].tick_params(axis='y', direction='in', right=True)
axes[1].tick_params(axis='y', direction='in', right=True)
plt.tight_layout()
plt.show()
#   Massaged (GNB ranker) ND test set
#       Discriminatory model vs massaged (GNB ranker) ND model
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#           Subplot accuracies
acc_d_nd_mas_gnb = [dt_d_nd_mas_gnb_acc, knn_d_nd_mas_gnb_acc, gnb_d_nd_mas_gnb_acc]
acc_nd_mas_gnb_nd_mas_gnb = [dt_nd_mas_gnb_nd_mas_gnb_acc, knn_nd_mas_gnb_nd_mas_gnb_acc, gnb_nd_mas_gnb_nd_mas_gnb_acc]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminatory model': acc_d_nd_mas_gnb,
               'Non-discriminatory model': acc_nd_mas_gnb_nd_mas_gnb}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classification algorithm',
           ylabel='Accuracy (%)', rot=0, ax=axes[0]))
#           Subplot discriminations
dscrm_d_nd_mas_gnb = [dt_d_nd_mas_gnb_dscrm, knn_d_nd_mas_gnb_dscrm, gnb_d_nd_mas_gnb_dscrm]
dscrm_nd_mas_gnb_nd_mas_gnb = [dt_nd_mas_gnb_nd_mas_gnb_dscrm, knn_nd_mas_gnb_nd_mas_gnb_dscrm,
                               gnb_nd_mas_gnb_nd_mas_gnb_dscrm]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminatory model': dscrm_d_nd_mas_gnb,
               'Non-discriminatory model': dscrm_nd_mas_gnb_nd_mas_gnb}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classification algorithm',
           ylabel='Discrimination (%)', rot=0, ax=axes[1]))

fig.suptitle('Accuracy and discrimination of the discriminatory and massaged (GNB ranker) non-discriminatory\n'
             'model for each algorithm on the massaged (GNB ranker) non-discriminatory test set')
axes[0].tick_params(axis='y', direction='in', right=True)
axes[1].tick_params(axis='y', direction='in', right=True)
plt.tight_layout()
plt.show()
#       Discriminatory model vs PS (GNB ranker) ND model
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#           Subplot accuracies
acc_d_nd_mas_gnb = [dt_d_nd_mas_gnb_acc, knn_d_nd_mas_gnb_acc, gnb_d_nd_mas_gnb_acc]
acc_nd_ps_gnb_nd_mas_gnb = [dt_nd_ps_gnb_nd_mas_gnb_acc, knn_nd_ps_gnb_nd_mas_gnb_acc, gnb_nd_ps_gnb_nd_mas_gnb_acc]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminatory model': acc_d_nd_mas_gnb,
               'Non-discriminatory model': acc_nd_ps_gnb_nd_mas_gnb}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classification algorithm',
           ylabel='Accuracy (%)', rot=0, ax=axes[0]))
#           Subplot discriminations
dscrm_d_nd_mas_gnb = [dt_d_nd_mas_gnb_dscrm, knn_d_nd_mas_gnb_dscrm, gnb_d_nd_mas_gnb_dscrm]
dscrm_nd_ps_gnb_nd_mas_gnb = [dt_nd_ps_gnb_nd_mas_gnb_dscrm, knn_nd_ps_gnb_nd_mas_gnb_dscrm,
                              gnb_nd_ps_gnb_nd_mas_gnb_dscrm]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminatory model': dscrm_d_nd_mas_gnb,
               'Non-discriminatory model': dscrm_nd_ps_gnb_nd_mas_gnb}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classification algorithm',
           ylabel='Discrimination (%)', rot=0, ax=axes[1]))

fig.suptitle('Accuracy and discrimination of the discriminatory and PS (GNB ranker) non-discriminatory\n'
             'model for each algorithm on the massaged (GNB ranker) non-discriminatory test set')
axes[0].tick_params(axis='y', direction='in', right=True)
axes[1].tick_params(axis='y', direction='in', right=True)
plt.tight_layout()
plt.show()

"""
#%% Visualizing results Nederlands (zelfde grafieken)
# Visualize accuracies & discriminations of models on discriminatory test set
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#   Subplot accuracies
acc_d_d = [dt_d_d_acc, knn_d_d_acc, gnb_d_d_acc]
acc_nd_d = [dt_nd_d_acc, knn_nd_d_acc, gnb_nd_d_acc]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminerend model': acc_d_d,
               'Non-discriminerend model': acc_nd_d}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classificatiealgoritme',
           ylabel='Accuraatheid (%)', rot=0, ax=axes[0]))
#   Subplot discriminations
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

# Visualize accuracies & discriminations of models on non-discriminatory test set
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#   Subplot accuracies
acc_d_nd = [dt_d_nd_acc, knn_d_nd_acc, gnb_d_nd_acc]
acc_nd_nd = [dt_nd_nd_acc, knn_nd_nd_acc, gnb_nd_nd_acc]
index = ['Decision tree', 'K-nearest neighbors', 'Gaussian Naive Bayes']
(pd.DataFrame({'Discriminerend model': acc_d_nd,
               'Non-discriminerend model': acc_nd_nd}, index)
 .plot.bar(yticks=[x / 10.0 for x in range(0, 11)], ylim=(0, 1), xlabel='Classificatiealgoritme',
           ylabel='Accuraatheid (%)', rot=0, ax=axes[0]))
#   Subplot discriminations
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
test = pd.concat([nd_ps_gnb_x_test, nd_ps_gnb_y_test], axis=1)
nd_train = pd.concat([nd_mas_gnb_x_train, nd_mas_gnb_y_train], axis=1)

a = (train.loc[(train.sex == True) & (train.income == True)].shape[0] / train.loc[train.sex == True].shape[0]) - (
        test.loc[(test.sex == True) & (test.income == True)].shape[0] / test.loc[test.sex == True].shape[0]) * (
            train.loc[train.sex == True].shape[0] / train.shape[0]) - (
            train.loc[(train.sex == False) & (train.income == True)].shape[0] /
            train.loc[train.sex == False].shape[0]) - (
            test.loc[(test.sex == False) & (test.income == True)].shape[0] / test.loc[test.sex == False].shape[0]) * (
            train.loc[train.sex == False].shape[0] / train.shape[0])
b = (nd_train.loc[(nd_train.sex == True) & (nd_train.income == True)].shape[0] /
     nd_train.loc[nd_train.sex == True].shape[0]) - (
            test.loc[(test.sex == True) & (test.income == True)].shape[0] / test.loc[test.sex == True].shape[0]) * (
            nd_train.loc[nd_train.sex == True].shape[0] / nd_train.shape[0]) - (
            nd_train.loc[(nd_train.sex == False) & (nd_train.income == True)].shape[0] /
            nd_train.loc[nd_train.sex == False].shape[0]) - (
            test.loc[(test.sex == False) & (test.income == True)].shape[0] / test.loc[test.sex == False].shape[0]) * (
            nd_train.loc[nd_train.sex == False].shape[0] / nd_train.shape[0])
print(a)
print(b)
