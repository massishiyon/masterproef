import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def normalize(feature, train_ind):
    var_train = feature.loc[train_ind]
    var_train_mean = np.mean(var_train)
    var_train_std = np.std(var_train)

    #Standardize variable
    var_preprocessed = (feature - var_train_mean) / var_train_std

    #Deal with outliers
    j = 0
    for i in var_preprocessed:
        if i > 3:
            var_preprocessed.iloc[j] = 3
        elif i < -3:
            var_preprocessed.iloc[j] = -3
        j += 1
    return var_preprocessed


def generate_dummies(feature, var_name):
    var_preprocessed = pd.get_dummies(feature, prefix=str(var_name))
    # leave the first category out, so it becomes the reference category
    var_preprocessed = var_preprocessed.iloc[:, 1:]
    return var_preprocessed


def train_and_test_clf(d_x_train, d_y_train, d_x_test, d_y_test, nd_mas_gnb_x_train, nd_mas_gnb_y_train, nd_mas_gnb_x_test, nd_mas_gnb_y_test,
                       nd_ps_gnb_x_train, nd_ps_gnb_y_train, nd_ps_gnb_x_test, nd_ps_gnb_y_test, algorithm,
                       d_clf=None):
    # Perform 5-fold cross validation grid search to find the optimal hyperparameters
    gridsearch = 0
    match algorithm:
        case "GNB" | "":
            param_grid = {
                'var_smoothing': np.logspace(0, -7)
            }
            gridsearch = GridSearchCV(GaussianNB(), param_grid, n_jobs=-1, verbose=1)
        case "DT":
            param_grid = {
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [20, 50, 100, 150],
                'min_samples_leaf': [10, 25, 50, 75],
                'max_features': [None, 'sqrt', 'log2'],
                'criterion': ['gini', 'entropy'],
            }
            gridsearch = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid, n_jobs=-1, verbose=1)
        case "KNN":
            param_grid = {
                'n_neighbors': [15, 20, 30, 40, 50],
                'weights': ['uniform', 'distance'],
                'leaf_size': [5, 10, 15]
            }
            gridsearch = GridSearchCV(KNeighborsClassifier(n_jobs=-1), param_grid, n_jobs=-1, verbose=1)
        case _:
            sys.exit("Typo in algorithm name")

    # If algorithm was not used as a ranker for removing discrimination from the training set, a model still needs to
    # be trained on the discriminatory (original) test set
    # If the algorithm was used as a ranker, then the discriminatory model has already been trained and should be
    # passed using the d_clf variable
    if d_clf is None:
        gridsearch.fit(d_x_train, d_y_train)
        print(gridsearch.best_params_)
        print(gridsearch.best_score_)
        d_clf = gridsearch.best_estimator_

    # Train non-discriminatory models on the non-discriminatory training sets
    #   Massaged (GNB ranker) ND model
    gridsearch.fit(nd_mas_gnb_x_train, nd_mas_gnb_y_train)
    print(gridsearch.best_params_)
    print(gridsearch.best_score_)
    nd_mas_gnb_clf = gridsearch.best_estimator_
    #   PS (GNB ranker) ND model
    gridsearch.fit(nd_ps_gnb_x_train, nd_ps_gnb_y_train)
    print(gridsearch.best_params_)
    print(gridsearch.best_score_)
    nd_ps_gnb_clf = gridsearch.best_estimator_

    # Compute accuracies (always compare discriminatory model vs any other model for a given set in graphs)
    #   Discriminatory test set (only for proving that accuracy will go down on this set, no need to use every model)
    #       Discriminatory model
    d_d_preds = d_clf.predict(d_x_test)
    d_d_acc = accuracy_score(d_y_test, d_d_preds)
    #       Massaged (GNB ranker) non-discriminatory model
    nd_d_preds = nd_mas_gnb_clf.predict(d_x_test)
    nd_d_acc = accuracy_score(d_y_test, nd_d_preds)
    #   PS (GNB ranker) non-discriminatory test set (proving that accuracy can go up)
    #       Discriminatory model
    d_nd_ps_gnb_preds = d_clf.predict(nd_ps_gnb_x_test)
    d_nd_ps_gnb_acc = accuracy_score(nd_ps_gnb_y_test, d_nd_ps_gnb_preds)
    #       Massaged (GNB ranker) non-discriminatory model
    nd_mas_gnb_nd_ps_gnb_preds = nd_mas_gnb_clf.predict(nd_ps_gnb_x_test)
    nd_mas_gnb_nd_ps_gnb_acc = accuracy_score(nd_ps_gnb_y_test, nd_mas_gnb_nd_ps_gnb_preds)
    #       PS (GNB ranker) non-discriminatory model
    nd_ps_gnb_nd_ps_gnb_preds = nd_ps_gnb_clf.predict(nd_ps_gnb_x_test)
    nd_ps_gnb_nd_ps_gnb_acc = accuracy_score(nd_ps_gnb_y_test, nd_ps_gnb_nd_ps_gnb_preds)
    #   Massaged (GNB ranker) non-discriminatory test set
    #       Discriminatory model
    d_nd_mas_gnb_preds = d_clf.predict(nd_mas_gnb_x_test)
    d_nd_mas_gnb_acc = accuracy_score(nd_mas_gnb_y_test, d_nd_mas_gnb_preds)
    #       Massaged (GNB ranker) non-discrminatory model
    nd_mas_gnb_nd_mas_gnb_preds = nd_mas_gnb_clf.predict(nd_ps_gnb_x_test)
    nd_mas_gnb_nd_mas_gnb_acc = accuracy_score(nd_ps_gnb_y_test, nd_mas_gnb_nd_mas_gnb_preds)
    #       PS (GNB ranker) non-discriminatory model
    nd_ps_gnb_nd_mas_gnb_preds = nd_ps_gnb_clf.predict(nd_ps_gnb_x_test)
    nd_ps_gnb_nd_mas_gnb_acc = accuracy_score(nd_ps_gnb_y_test, nd_ps_gnb_nd_mas_gnb_preds)

    # Compute statistical parity discriminations
    #   Discriminatory test set (again proving the hypothesis from Kamiran & Calders)
    #       Discriminatory model
    test_preds = pd.DataFrame(d_x_test.loc[:, 'sex'])
    test_preds.loc[:, 'prediction'] = d_d_preds
    d_d_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                     test_preds.loc[test_preds.sex == True].shape[0]) - (
                            test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                            test_preds.loc[test_preds.sex == False].shape[0])
    #       Massaged (GNB ranker) non-discriminatory model
    test_preds = pd.DataFrame(d_x_test.loc[:, 'sex'])
    test_preds.loc[:, 'prediction'] = nd_d_preds
    nd_d_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                      test_preds.loc[test_preds.sex == True].shape[0]) - (
                             test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[0] /
                             test_preds.loc[test_preds.sex == False].shape[0])
    #   PS (GNB ranker) non-discriminatory test set (proving own hypothesis)
    #       Discriminatory model
    test_preds = pd.DataFrame(nd_ps_gnb_x_test.loc[:, 'sex'])
    test_preds.loc[:, 'prediction'] = d_nd_ps_gnb_preds
    d_nd_ps_gnb_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                             test_preds.loc[test_preds.sex == True].shape[0]) - (
                                    test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[
                                        0] /
                                    test_preds.loc[test_preds.sex == False].shape[0])
    #       Massaged (GNB ranker) non-discriminatory model
    test_preds = pd.DataFrame(nd_ps_gnb_x_test.loc[:, 'sex'])
    test_preds.loc[:, 'prediction'] = nd_mas_gnb_nd_ps_gnb_preds
    nd_mas_gnb_nd_ps_gnb_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[
                                          0] /
                                      test_preds.loc[test_preds.sex == True].shape[0]) - (
                                             test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)]
                                             .shape[0] / test_preds.loc[test_preds.sex == False].shape[0])
    #       PS (GNB ranker) non-discriminatory model
    test_preds = pd.DataFrame(nd_ps_gnb_x_test.loc[:, 'sex'])
    test_preds.loc[:, 'prediction'] = nd_ps_gnb_nd_ps_gnb_preds
    nd_ps_gnb_nd_ps_gnb_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[
                                         0] /
                                     test_preds.loc[test_preds.sex == True].shape[0]) - (
                                            test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)]
                                            .shape[0] / test_preds.loc[test_preds.sex == False].shape[0])
    #   Massaged (GNB ranker) non-discriminatory test set (proving own hypothesis)
    #       Discriminatory model
    test_preds = pd.DataFrame(nd_mas_gnb_x_test.loc[:, 'sex'])
    test_preds.loc[:, 'prediction'] = d_nd_mas_gnb_preds
    d_nd_mas_gnb_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[0] /
                              test_preds.loc[test_preds.sex == True].shape[0]) - (
                                     test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)].shape[
                                         0] /
                                     test_preds.loc[test_preds.sex == False].shape[0])
    #       Massaged (GNB ranker) non-discriminatory model
    test_preds = pd.DataFrame(nd_mas_gnb_x_test.loc[:, 'sex'])
    test_preds.loc[:, 'prediction'] = nd_mas_gnb_nd_mas_gnb_preds
    nd_mas_gnb_nd_mas_gnb_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[
                                           0] /
                                       test_preds.loc[test_preds.sex == True].shape[0]) - (
                                              test_preds.loc[
                                                  (test_preds.sex == False) & (test_preds.prediction == True)]
                                              .shape[0] / test_preds.loc[test_preds.sex == False].shape[0])
    #       PS (GNB ranker) non-discriminatory model
    test_preds = pd.DataFrame(nd_mas_gnb_x_test.loc[:, 'sex'])
    test_preds.loc[:, 'prediction'] = nd_ps_gnb_nd_mas_gnb_preds
    nd_ps_gnb_nd_mas_gnb_dscrm = (test_preds.loc[(test_preds.sex == True) & (test_preds.prediction == True)].shape[
                                          0] /
                                      test_preds.loc[test_preds.sex == True].shape[0]) - (
                                             test_preds.loc[(test_preds.sex == False) & (test_preds.prediction == True)]
                                             .shape[0] / test_preds.loc[test_preds.sex == False].shape[0])

    # Print accuracies & discriminations
    #   Discriminatory test set
    #       Discriminatory model
    print("Accuracy of the discriminatory " + algorithm + " model on the discriminatory test set: " + str(d_d_acc))
    print("Discrimination of the discriminatory " + algorithm + " model on the discriminatory test set: " + str(d_d_dscrm))
    #       Massaged (GNB ranker) non-discriminatory model
    print("Accuracy of the massaged (GNB ranker) non-discriminatory " + algorithm + " model on the discriminatory test set: " + str(
        nd_d_acc))
    print(
        "Discrimination of the massaged (GNB ranker) non-discriminatory " + algorithm + " model on the discriminatory test set: " + str(
            nd_d_dscrm))
    #   PS (GNB ranker) non-discriminatory test set
    #       Discriminatory model
    print("Accuracy of the discriminatory " + algorithm + " model on the PS (GNB ranker) non-discriminatory test set: " + str(
        d_nd_ps_gnb_acc))
    print(
        "Discrimination of the discriminatory " + algorithm + " model on the PS (GNB ranker) non-discriminatory test set: " + str(
            d_nd_ps_gnb_dscrm))
    #       Massaged (GNB ranker) non-discriminatory model
    print(
        "Accuracy of the massaged (GNB ranker) non-discriminatory " + algorithm + " model on the PS (GNB ranker) non-discriminatory test set: " +
        str(nd_mas_gnb_nd_ps_gnb_acc))
    print(
        "Discrimination of the massaged (GNB ranker) non-discriminatory " + algorithm + " model on the PS (GNB ranker) non-discriminatory test set: " +
        str(nd_mas_gnb_nd_ps_gnb_dscrm))
    #       PS (GNB ranker) non-discriminatory model
    print(
        "Accuracy of the PS (GNB ranker) non-discriminatory " + algorithm + " model on the PS (GNB ranker) non-discriminatory test set: " +
        str(nd_ps_gnb_nd_ps_gnb_acc))
    print(
        "Discrimination of the PS (GNB ranker) non-discriminatory " + algorithm + " model on the PS (GNB ranker) non-discriminatory test set: " +
        str(nd_ps_gnb_nd_ps_gnb_dscrm))
    #   Massaged (GNB ranker) non-discriminatory test set
    #       Discriminatory model
    print("Accuracy of the discriminatory " + algorithm + " model on the Massaged (GNB ranker) non-discriminatory test set: " + str(
        d_nd_mas_gnb_acc))
    print(
        "Discrimination of the discriminatory " + algorithm + " model on the Massaged (GNB ranker) non-discriminatory test set: " + str(
            d_nd_mas_gnb_dscrm))
    #       Massaged (GNB ranker) non-discriminatory model
    print(
        "Accuracy of the massaged (GNB ranker) non-discriminatory " + algorithm + " model on the Massaged (GNB ranker) non-discriminatory test set: " +
        str(nd_mas_gnb_nd_mas_gnb_acc))
    print(
        "Discrimination of the massaged (GNB ranker) non-discriminatory " + algorithm + " model on the Massaged (GNB ranker) non-discriminatory test set: " +
        str(nd_mas_gnb_nd_mas_gnb_dscrm))
    #       PS (GNB ranker) non-discriminatory model
    print(
        "Accuracy of the PS (GNB ranker) non-discriminatory " + algorithm + " model on the Massaged (GNB ranker) non-discriminatory test set: " +
        str(nd_ps_gnb_nd_mas_gnb_acc))
    print(
        "Discrimination of the PS (GNB ranker) non-discriminatory " + algorithm + " model on the Massaged (GNB ranker) non-discriminatory test set: " +
        str(nd_ps_gnb_nd_mas_gnb_dscrm))

    return (d_d_acc, nd_d_acc, d_nd_ps_gnb_acc, nd_mas_gnb_nd_ps_gnb_acc, nd_ps_gnb_nd_ps_gnb_acc, d_nd_mas_gnb_acc,
            nd_mas_gnb_nd_mas_gnb_acc, nd_ps_gnb_nd_mas_gnb_acc, d_d_dscrm, nd_d_dscrm, d_nd_ps_gnb_dscrm,
            nd_mas_gnb_nd_ps_gnb_dscrm, nd_ps_gnb_nd_ps_gnb_dscrm, d_nd_mas_gnb_dscrm, nd_mas_gnb_nd_mas_gnb_dscrm,
            nd_ps_gnb_nd_mas_gnb_dscrm)
