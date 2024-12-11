import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import sys


# Still specific to census income dataset!
def massaging(df, indices_set, d_x_set, d_y_set, algorithm):
    # Before massaging, calculate chance of having income >50K given each sex and discrimination
    print("Before massaging:")
    calc_dscrmn(df, indices_set)

    # Perform 5-fold cross validation grid search to find the optimal hyperparameters
    gridsearch = 0
    match algorithm:
        case "GNB" | "":
            # Or: case algorithm if algorithm in ["GNB", ""]:
            param_grid = {
                'var_smoothing': np.logspace(0, -7)
            }
            gridsearch = GridSearchCV(GaussianNB(), param_grid, n_jobs=-1, verbose=1)
        case "DT":
            ""
        case "KNN":
            ""
        case _:
            sys.exit("Typo in algorithm name")

    gridsearch.fit(d_x_set, d_y_set)

    # Results of grid search
    print(gridsearch.best_params_)
    print(gridsearch.best_score_)
    clf = gridsearch.best_estimator_

    # Predict probability scores on set
    scores = clf.predict_proba(d_x_set)[:, 1]

    # Create dataframes of candidates for promotion and demotion
    # Promotion candidates: females with low income
    # Demotion candidates: males with high income
    # set_scores = pd.concat([df.loc[indices_set, ['sex', 'income']], pd.Series(scores, name='score')], axis=1)
    set_scores = df.loc[indices_set, ['sex', 'income']]
    set_scores.loc[:, 'score'] = scores

    promotion_candidates = set_scores.loc[(set_scores.sex == False) & (set_scores.income == False)].sort_values(
        by='score', ascending=False)
    demotion_candidates = set_scores.loc[(set_scores.sex == True) & (set_scores.income == True)].sort_values(
        by='score')

    # As long as the statistical parity discrimination of set is bigger than 0, keep iterating
    print('Starting promotions/demotions...')
    i = 0
    while (df.loc[indices_set].loc[(df.sex == True) & (df.income == True)].shape[0] /
           df.loc[indices_set].loc[df.sex == True].shape[0]) - (
            df.loc[indices_set].loc[(df.sex == False) & (df.income == True)].shape[0] /
            df.loc[indices_set].loc[df.sex == False].shape[0]) > 0:
        df.loc[promotion_candidates.index[i], 'income'] = True
        df.loc[demotion_candidates.index[i], 'income'] = False
        i += 1
    print("Amount of promotions/demotions = " + str(i))

    # Save features X and outcome Y of the updated non-discriminatory set in their own dataframes
    nd_x_set = df.loc[indices_set].drop('income', axis=1)
    nd_y_set = df.loc[indices_set, 'income']

    # After massaging, calculate chance of having income >50K given each sex and discrimination
    print("After massaging:")
    calc_dscrmn(df, indices_set)

    return nd_x_set, nd_y_set, clf


def preferential_sampling(df, indices_set, d_x_set, d_y_set, algorithm):
    # Before preferential sampling, calculate chance of having income >50K given each sex and discrimination
    print("Before preferential sampling:")
    calc_dscrmn(df, indices_set)

    # Perform 5-fold cross validation grid search to find the optimal hyperparameters
    gridsearch = 0
    match algorithm:
        case "GNB" | "":
            param_grid = {
                'var_smoothing': np.logspace(0, -7)
            }
            gridsearch = GridSearchCV(GaussianNB(), param_grid, n_jobs=-1, verbose=1)
        case "DT":
            ""
        case "KNN":
            ""
        case _:
            sys.exit("Typo in algorithm name")

    gridsearch.fit(d_x_set, d_y_set)

    # Results of grid search
    print(gridsearch.best_params_)
    print(gridsearch.best_score_)
    clf = gridsearch.best_estimator_

    # Predict probability scores on set
    scores = clf.predict_proba(d_x_set)[:, 1]

    # Concatenate dataframe with probability scores
    # set_scores = pd.concat([df.loc[indices_set, ['sex', 'income']], pd.Series(scores, name='score')], axis=1)
    set_scores = df.loc[indices_set, ['sex', 'income']]
    set_scores.loc[:, 'score'] = scores

    # calculate expected size of each combination of socio-demographic and class
    # DN = Deprived community with negative class
    # FP = Favored community with positive class
    DN_size = int(round((set_scores.loc[(set_scores.sex == False)].shape[0] *
                         set_scores.loc[(set_scores.income == False)].shape[0]) / set_scores.shape[0]))
    DP_size = int(round((set_scores.loc[(set_scores.sex == False)].shape[0] *
                         set_scores.loc[(set_scores.income == True)].shape[0]) / set_scores.shape[0]))
    FN_size = int(round((set_scores.loc[(set_scores.sex == True)].shape[0] *
                         set_scores.loc[(set_scores.income == False)].shape[0]) / set_scores.shape[0]))
    FP_size = int(round((set_scores.loc[(set_scores.sex == True)].shape[0] *
                         set_scores.loc[(set_scores.income == True)].shape[0]) / set_scores.shape[0]))

    # Take a subset of each combination of socio-demographic and class with probability scores and sort by score,
    # sort the subsets concerning the negative class descending
    DN = set_scores.loc[(set_scores.sex == False) & (set_scores.income == False)].sort_values(by='score',
                                                                                              ascending=False)
    DP = set_scores.loc[(set_scores.sex == False) & (set_scores.income == True)].sort_values(by='score')
    FN = set_scores.loc[(set_scores.sex == True) & (set_scores.income == False)].sort_values(by='score',
                                                                                             ascending=False)
    FP = set_scores.loc[(set_scores.sex == True) & (set_scores.income == True)].sort_values(by='score')

    # Print expected and actual sizes
    print("DN expected size: " + str(DN_size))
    print('DN actual size: ' + str(DN.shape[0]))
    print("DP expected size: " + str(DP_size))
    print('DP actual size: ' + str(DP.shape[0]))
    print("FN expected size: " + str(FN_size))
    print('FN actual size: ' + str(FN.shape[0]))
    print("FP expected size: " + str(FP_size))
    print('FP actual size: ' + str(FP.shape[0]))

    # As long as the observed size of the combination of the socio-demographic and the class in the set is
    # smaller or larger than the expected size, duplicate or remove samples
    print("Starting DN removals...")
    i = 0
    while df.loc[indices_set].loc[(df.sex == False) & (df.income == False)].shape[0] > DN_size:
        # Remove record from dataframe
        df = df.drop(DN.index[i])
        # Remove record's index from set indices
        indices_set = np.delete(indices_set, np.argwhere(indices_set == DN.index[i]))
        i += 1
    print(str(i) + ' DN removals')

    print("Starting DP duplications...")
    i = 0
    while df.loc[indices_set].loc[(df.sex == False) & (df.income == True)].shape[0] < DP_size:
        # Duplicate record in dataframe
        row = df.loc[DP.index[i]]
        row.name = np.amax(df.index) + 1
        df = pd.concat([df, pd.DataFrame([row])])
        # df.loc[df.index[-1]+1] = df.loc[DP.index[0]]
        # Add index of duplicated record to set indices
        indices_set = np.append(indices_set, np.amax(df.index))
        i += 1
        # When the amount of records that need to be duplicated exceeds the amount of records with the combination of
        # the socio-demographic and the class, simply reiterate over the subset when the end is reached
        if i >= DP.shape[0]:
            i = 0
    print(str(i) + ' DP duplications')

    print("Starting FN duplications...")
    i = 0
    while df.loc[indices_set].loc[(df.sex == True) & (df.income == False)].shape[0] < FN_size:
        # Duplicate record in dataframe
        row = df.loc[FN.index[i]]
        row.name = np.amax(df.index) + 1
        df = pd.concat([df, pd.DataFrame([row])])
        # Add index of duplicated record to set indices
        indices_set = np.append(indices_set, np.amax(df.index))
        i += 1
        # When the amount of records that need to be duplicated exceeds the amount of records with the combination of
        # the socio-demographic and the class, simply reiterate over the subset when the end is reached
        if i >= FN.shape[0]:
            i = 0
    print(str(i) + ' FN duplications')

    print("Starting FP removals...")
    i = 0
    while df.loc[indices_set].loc[(df.sex == True) & (df.income == True)].shape[0] > FP_size:
        # Remove record from dataframe
        df = df.drop(FP.index[i])
        # Remove record's index from set indices
        indices_set = np.delete(indices_set, np.argwhere(indices_set == FP.index[i]))
        i += 1
    print(str(i) + ' FP removals')

    # Save features X and outcome Y of the updated non-discriminatory set in their own dataframes
    nd_x_set = df.loc[indices_set].drop('income', axis=1)
    nd_y_set = df.loc[indices_set, 'income']

    # After preferential sampling, calculate chance of having income >50K given each sex and discrimination
    print("After preferential sampling:")
    calc_dscrmn(df, indices_set)

    return nd_x_set, nd_y_set, clf


def calc_dscrmn(df, indices_set):
    # Chance of having income >50K given being male for set
    highincome_male_prob = df.loc[indices_set].loc[(df.sex == True) & (df.income == True)].shape[0] / \
                                df.loc[indices_set].loc[df.sex == True].shape[0]
    print("Chance of having income >50K given being male: " + str("{:.8f}".format(highincome_male_prob)))
    # Chance of having income >50K given being female for set
    highincome_female_prob = df.loc[indices_set].loc[(df.sex == False) & (df.income == True)].shape[0] / \
                                  df.loc[indices_set].loc[df.sex == False].shape[0]
    print("Chance of having income >50K given being female: " + str("{:.8f}".format(highincome_female_prob)))
    # Discrimination for set
    discr = highincome_male_prob - highincome_female_prob
    print("Statistical parity discrimination: " + str("{:.8f}".format(discr)))
