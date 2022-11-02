import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from os.path import exists
pd.options.mode.chained_assignment = None

NUM_SPLITS = 5
TRAIN_SIZE = 2/3
BIN_SIZES = [5, 10, 15, 20]
CLASS = 'class'
SETOSA = 'Iris-setosa'
NOTSETOSA = 'Iris-versicolor'


def count(data, colname, bin_num, target):
    condition = (data[colname] == bin_num) & (data['class'] == target)
    return len(data[condition])

def create_splits():
    data = pd.read_csv("../data/iris.csv")
    data[CLASS] = data[CLASS].replace('Iris-virginica', 'Iris-versicolor')
    
    for split_num in range(NUM_SPLITS):
        data = data.sample(frac=1) # shuffle the data

        for bin_sz in BIN_SIZES:
            data_copy = data.copy(deep=True)

            # split into test and train
            train_len = int(TRAIN_SIZE*len(data_copy))
            train = data_copy.iloc[:train_len,:]
            train.to_csv("../data/bin_"+ str(bin_sz) + "/train/split_" + str(split_num) + ".csv", index=False)
            test = data_copy.iloc[train_len:,:]
            test.to_csv("../data/bin_"+ str(bin_sz)+"/test/split_" + str(split_num) + ".csv", index=False)
            
def get_probabilities():
    accuracies = {bin_s: [] for bin_s in BIN_SIZES}
    F1_measure = {bin_s: [] for bin_s in BIN_SIZES}
    FPR = {bin_s: [] for bin_s in BIN_SIZES}
    TPR = {bin_s: [] for bin_s in BIN_SIZES}
    
    for split_num in range(NUM_SPLITS):
        print("splitnum: ", split_num)
        roc = {bin_s: [[],[]] for bin_s in BIN_SIZES}
        for bin_sz in BIN_SIZES:
            print("binsize: ", bin_sz)
            train_data = pd.read_csv("../data/bin_" + str(bin_sz) + "/train/split_" + str(split_num) + ".csv", index_col=False)
            test_data = pd.read_csv("../data/bin_" + str(bin_sz) + "/test/split_" + str(split_num) + ".csv", index_col=False)
            prediction_data = test_data.copy(deep=True)

            # discretize the data
            for col in train_data.columns[:-1]:
                # avg = mean(train_data[col], test_data[col])
                # train_data[col] = train_data[col].replace(0, avg)
                train_data[col] = pd.cut(train_data[col], bins=bin_sz, labels=False)
                test_data[col] = pd.cut(test_data[col], bins=bin_sz, labels=False)

            # calculate train probabilities
            n_yes = count(train_data, CLASS, SETOSA, SETOSA)
            n_no = count(train_data, CLASS, NOTSETOSA, NOTSETOSA)
            train_probabilities = {SETOSA:{}, NOTSETOSA:{}}

            for column in train_data.columns[:-1]:
                train_probabilities[SETOSA][column] = {}
                train_probabilities[NOTSETOSA][column] = {}

                for i in range(bin_sz):
                    nc_yes = count(train_data, column, i, SETOSA)
                    nc_no = count(train_data, column, i, NOTSETOSA)

                    train_probabilities[SETOSA][column][i] = (nc_yes + 1)/(n_yes + bin_sz)
                    train_probabilities[NOTSETOSA][column][i] = (nc_no+ 1)/(n_no + bin_sz)
            
            true_pos, true_neg, false_pos,false_neg = 0,0,0,0
            train_prob_setosa = n_yes / len(train_data)
            train_prob_not_setosa = n_no / len(train_data)

            min_diff = 2.0
            max_diff = 0.0

            # Calculate F-measures on test data
            for row in range(len(test_data)):
                prod_setosa = train_prob_setosa
                prod_not_setosa = train_prob_not_setosa

                for feature in test_data.columns[:-1]:
                    prod_setosa *= train_probabilities[SETOSA][feature][test_data[feature].iloc[row]]
                    prod_not_setosa *= train_probabilities[NOTSETOSA][feature][test_data[feature].iloc[row]]
            
                min_diff = min(abs(prod_setosa - prod_not_setosa), min_diff)
                max_diff = max(abs(prod_setosa - prod_not_setosa), max_diff)

                if prod_setosa > prod_not_setosa:
                    prediction_data[CLASS].iloc[row] = SETOSA
                    if test_data[CLASS].iloc[row] == SETOSA:
                        true_pos += 1
                    else:
                        false_pos += 1
                else:
                    prediction_data[CLASS].iloc[row] = NOTSETOSA
                    if test_data[CLASS].iloc[row] == NOTSETOSA:
                        true_neg += 1
                    else:
                        false_neg += 1
            
            prediction_data.to_csv("../data/bin_"+ str(bin_sz)+"/predictions/split_" + str(split_num) + ".csv", index=False)

            accuracies[bin_sz].append((true_pos + true_neg) / len(test_data))
            print("true pos: ", true_pos)
            print("false pos: ", false_pos)
            print("true neg: ", true_neg)
            print("false neg: ", false_neg)
            # Calculate the F1 scores
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)

            F1_measure[bin_sz].append((2*precision*recall)/(precision+recall))
            TPR[bin_sz].append(true_pos / (true_pos + false_neg))
            FPR[bin_sz].append(false_pos/ (true_neg + false_pos))


            # Construct ROC on test data
            for threshold in reversed(np.linspace(min_diff
            ,max_diff,10)):
                tp, fp = 0,0
                for row in range(len(test_data)):
                    prod_setosa = train_prob_setosa
                    prod_not_setosa = train_prob_not_setosa
                    
                    for feature in test_data.columns[:-1]:
                        prod_setosa *= train_probabilities[SETOSA][feature][test_data[feature].iloc[row]]
                        prod_not_setosa *= train_probabilities[NOTSETOSA][feature][test_data[feature].iloc[row]]

                    if abs(prod_setosa - prod_not_setosa) >= threshold:
                        if test_data[CLASS].iloc[row] == SETOSA:
                            tp += 1
                        else:
                            fp += 1
                tpr = tp/true_pos
                fpr = fp/true_neg
                roc[bin_sz][0].append(fpr)
                roc[bin_sz][1].append(tpr)
        # End bin size for-loop

        fig = plt.figure()
        for i, key in enumerate(roc):
            plt.plot(roc[key][0], roc[key][1], label=key, linestyle='--', dashes=(5, i))
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC of Setosa vs. Not Setosa')
            plt.legend()
        fig.savefig("../figures/roc_curve_plot_" + str(split_num) + ".png")
        plt.clf()
    # End split_num for-loop


    fig = plt.figure()
    for i, key in enumerate(accuracies):
        plt.plot(range(NUM_SPLITS), accuracies[key], label=str(key) + " bins:", linestyle="--", dashes=(5, i))
        plt.xlabel('Round Number')
        plt.ylabel('Accuracy Score')
        plt.xticks(range(NUM_SPLITS))
        plt.title('Plot of Accuracy Scores')
        plt.legend()

        print(str(key) + " bins:")
        print("\tAccuracies: " + str(accuracies[key]))   
        print("\tMin Acc: " + str(min(accuracies[key])))
        print("\tMax Acc: " + str(max(accuracies[key])))
        print("\tAvg Acc: " + str(mean(accuracies[key])))

    fig.savefig("../figures/accuracy_score_plot.png")
    plt.clf()
    
    fig6 = plt.figure()
    for i, key in enumerate(F1_measure):
        plt.plot(range(NUM_SPLITS), F1_measure[key], label=str(key) + " bins:", linestyle='--', dashes=(5, i))
        plt.xlabel('Round Number')
        plt.ylabel('F1 Scores')
        plt.xticks(range(NUM_SPLITS))
        
        plt.title('Plot of F1 Scores')
        plt.legend()
    fig6.savefig("../figures/f1_score_plot.png")
    plt.clf()


def get_input(min_val: int, max_val: int, prompt: str) -> int:
    """Loops through prompt until user enters a valid value

    Args:
        min_val (`int`): Minimum inlcusive valid value
        max_val (`int`): Maximum inclusive valid value
        prompt (`str`): The string prompt to be asked at input
    
    Returns:
        `int`: The user's chosen value
    """
    while True:
        try:
            n = int(input(prompt))
            if n > max_val or n < min_val:
                print("error!")
                continue
            else:
                break
        except:
            print("error!")
    return n

def missing_data():
    for split_num in range(NUM_SPLITS):
        for bin_sz in BIN_SIZES:
            if not exists("../data/bin_" + str(bin_sz) + "/train/split_" + str(split_num) + ".csv"):
                return(True)
            if not exists("../data/bin_" + str(bin_sz) + "/test/split_" + str(split_num) + ".csv"):
                return(True)
    return False


def interpret_intput(n: int) -> int:
    """Runs function corresponding to the user's choice from main menu 

    Args:
        n (`int`): The number corresponding to the user's choice

    Returns:
        `int`: Returns 0 to loop through main menu again and -1 to exit the loop
    """
    if n == 1:
        create_splits()
        print("\nSplit data can be found in data/bin_#/test and data/bin_#/train\n")
        return 0
    elif n == 2:
        if missing_data():
            print("Either create the train-test splits on Iris data set with function (1) or make sure you are not missing any files in each bin's train and test folders\n")
        else:
            get_probabilities()
        return 0
    else:
        return -1
import os
def main():
    print("Assignment 3: Naive Bayes")
    # path = "../data/bin_20/predictions/"
    # for filename in os.listdir(path):
    #     data = pd.read_csv(path + filename, usecols=range(1,6))
    #     data.columns = ['sepallength','sepalwidth','petallength', 'petalwidth', 'class']
    #     data.to_csv(path + filename, index=False)

    while True:
        n = get_input(1,3, "(1) Create train-test splits on Iris data set\n(2) Train model using train-test data to from data/bin_#/train and data/bin_#/test\n(3) Quit\nChoose number to run function: ")
        
        status = interpret_intput(n)
        if status == -1:
            print("Thank You!")
            break                 

if __name__ == "__main__":
    main()