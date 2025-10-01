#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree
import utils

"""
python code/main.py 1
"""
@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = [1, 3, 10]

    for k in ks:
        knn = KNN(k)
        knn.fit(X, y)

        train_predict = knn.predict(X)
        test_predict = knn.predict(X_test)

        train_acc = sum((train_predict==y)/len(train_predict))
        test_acc = sum((test_predict==y_test)/len(test_predict))

        print("Train accuracty for k = {} is {}".format(k, train_acc))
        print("Test accuracy for k = {} is {}".format(k, test_acc))

    knn = KNN(1)
    knn.fit(X, y)
    utils.plot_classifier(knn, X, y)

"""
python code/main.py 2

"""
@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    num_train = len(X)
    num_val = int(num_train / 10)

    ks = list(range(1, 30, 4))
    cv_accs = []
    cv_train_accs = []
    for k in ks:
        print(k)
        train_accs = []
        val_accs = []

        for i in range(10):
            print(i)
            val_mask = np.full(num_train, False)
            val_mask[i * num_val: (i + 1) * num_val] = True
            
            cur_val_X = X[val_mask]
            cur_val_y = y[val_mask]

            cur_train_X = X[~val_mask]
            cur_train_y = y[~val_mask]

            knn = KNN(k)
            knn.fit(cur_train_X, cur_train_y)

            
            train_predict = knn.predict(cur_train_X)
            val_predict = knn.predict(cur_val_X)


            train_acc = sum((train_predict==cur_train_y)/len(train_predict))
            val_acc = sum((val_predict==cur_val_y)/len(val_predict))

            train_accs.append(train_acc)
            val_accs.append(val_acc)
        
        cv_accs.append(np.mean(val_accs))
        cv_train_accs.append(np.mean(train_accs))

    
    breakpoint()
    raise NotImplementedError()


# [0.6775000000000004, 0.7075000000000007, 0.7145000000000005, 0.7265000000000006, 0.7355000000000006, 0.7305000000000005, 0.7265000000000006, 0.7220000000000005]

# [0.9999999999999684, 0.8053888888888657, 0.7806666666666445, 0.7729999999999783, 0.7718333333333116, 0.7652222222222008, 0.7603888888888677, 0.7563888888888679]

@handle("2.1")
def q21():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print("2.1")
    cv_accs = [0.6775000000000004, 0.7075000000000007, 0.7145000000000005, 0.7265000000000006, 0.7355000000000006, 0.7305000000000005, 0.7265000000000006, 0.7220000000000005]
    test_accs = []

    ks = list(range(1, 30, 4))

    for k in ks:
        print(k)
        knn = KNN(k)
        knn.fit(X, y)
        
        test_predict = knn.predict(X_test)
        test_acc = sum((test_predict==y_test)/len(test_predict))
        test_accs.append(test_acc)


    print(test_accs)

    plt.scatter(ks, test_accs, label="Test Accuracy")
    plt.scatter(ks, cv_accs, label="CV Accuracy")
    plt.xlabel("K-Value")
    plt.ylabel("Accuracy")
    plt.title("KNN: CV vs. Test Accuracy")

    plt.legend()
    plt.savefig("../figs/Accuracy")




@handle("2.2")
def q22():
    train_acc = [0.9999999999999684, 0.8053888888888657, 0.7806666666666445, 0.7729999999999783, 0.7718333333333116, 0.7652222222222008, 0.7603888888888677, 0.7563888888888679]
    ks = list(range(1, 30, 4))


    train_acc = [1-a for a in train_acc]


    plt.scatter(ks, train_acc, label="Train Error")
    plt.xlabel("K-Value")
    plt.ylabel("Error Rate")
    plt.title("KNN: Train Error")

    plt.legend()
    plt.savefig("../figs/Error")


@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    """YOUR CODE HERE FOR Q3.2"""
    raise NotImplementedError()



@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    """CODE FOR Q3.4: Modify naive_bayes.py/NaiveBayesLaplace"""

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    """YOUR CODE HERE FOR Q3.4. Also modify naive_bayes.py/NaiveBayesLaplace"""
    raise NotImplementedError()



@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    """YOUR CODE FOR Q4. Also modify random_tree.py/RandomForest"""
    raise NotImplementedError()



@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.1. Also modify kmeans.py/Kmeans"""
    raise NotImplementedError()



@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.2"""
    raise NotImplementedError()



if __name__ == "__main__":
    main()
