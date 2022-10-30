from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from statistics import mean

import constant as constant
        
class Labels(str, Enum):
    SETOSA = "setosa"
    NOTSETOSA = "not-setosa"
    CLASS = "class"

def count(data, colname, bin_num, target):
    condition = (data[colname] == bin_num) & (data[Labels.CLASS] == target)
    return len(data[condition])

def main() -> None:
    data = pd.read_csv("../data/iris.csv")
    accuracies = {'bin_5': [], 'bin_10': [], 'bin_15': [], 'bin_20': []}
    F1_measure = {'bin_5': [], 'bin_10': [], 'bin_15': [], 'bin_20': []}
    FPR = {'bin_5': [], 'bin_10': [], 'bin_15': [], 'bin_20': []}
    TPR = {'bin_5': [], 'bin_10': [], 'bin_15': [], 'bin_20': []}
    
    
    for split_num in range(constant.NUM_SPLITS):
        data = data.sample(frac=1) # shuffle the data
        data_copy = data.copy(deep=True) # not really sure if this line is necessary
        roc = {'bin_5': [[],[]], 'bin_10': [[],[]], 'bin_15': [[],[]], 'bin_20': [[],[]]}
        for bin_sz in constant.BIN_SIZES:
            data_copy = data.copy(deep=True)

            # discretize the data
            for j in data_copy.columns[:-1]:
                avg = mean(data_copy[j])
                data_copy[j] = data_copy[j].replace(0, avg)
                data_copy[j] = pd.cut(data_copy[j], bins=bin_sz, labels=False)

            # split into test and train
            train_len = int(constant.TRAIN_SIZE*len(data_copy))
            train = data_copy.iloc[:train_len,:]
            test = data_copy.iloc[train_len:,:]
            
            # calculate train probabilities
            n_yes = count(train, Labels.CLASS, Labels.SETOSA, Labels.SETOSA)
            n_no = count(train, Labels.CLASS, Labels.NOTSETOSA, Labels.NOTSETOSA)
            train_probabilities = {Labels.SETOSA:{}, Labels.NOTSETOSA:{}}

            for column in train.columns[:-1]:
                train_probabilities[Labels.SETOSA][column] = {}
                train_probabilities[Labels.NOTSETOSA][column] = {}

                for i in range(bin_sz):
                    nc_yes = count(train, column, i, Labels.SETOSA)
                    nc_no = count(train, column, i, Labels.NOTSETOSA)

                    train_probabilities[Labels.SETOSA][column][i] = (nc_yes + 1)/(n_yes + bin_sz)
                    train_probabilities[Labels.NOTSETOSA][column][i] = (nc_no+ 1)/(n_no + bin_sz)
            
            true_pos, true_neg, false_pos,false_neg = 0,0,0,0
            train_prob_setosa = n_yes / len(train)
            train_prob_not_setosa = n_no / len(train)

            min_diff = 2.0
            max_diff = 0.0

            # Calculate F-measures on test data
            for row in range(len(test)):
                prod_setosa = train_prob_setosa
                prod_not_setosa = train_prob_not_setosa

                for feature in test.columns[:-1]:
                    prod_setosa *= train_probabilities[Labels.SETOSA][feature][test[feature].iloc[row]]
                    prod_not_setosa *= train_probabilities[Labels.NOTSETOSA][feature][test[feature].iloc[row]]
            
                min_diff = min(abs(prod_setosa - prod_not_setosa), min_diff)
                max_diff = max(abs(prod_setosa - prod_not_setosa), max_diff)
                if prod_setosa > prod_not_setosa:
                    if test[Labels.CLASS].iloc[row] == Labels.SETOSA:
                        true_pos += 1
                    else:
                        false_pos += 1
                else:
                    if test[Labels.CLASS].iloc[row] == Labels.NOTSETOSA:
                        true_neg += 1
                    else:
                        false_neg += 1
            
            bin_string = "bin_"+str(bin_sz)
            accuracies[bin_string].append((true_pos + true_neg) / len(test))

            # Calculate the F1 scores
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)

            F1_measure[bin_string].append((2*precision*recall)/(precision+recall))
            TPR[bin_string].append(true_pos / (true_pos + false_neg))
            FPR[bin_string].append(false_pos/ (true_neg + false_pos))

            # print("tp: ", true_pos)
            # print("fp: ", false_pos)
            # print("tn: ", true_neg)
            # print("fn: ", false_neg)
            # print(train_probabilities)

            # Construct ROC on test data

            for threshold in reversed(np.linspace(min_diff
            ,max_diff,10)):
                tp, fp = 0,0
                for row in range(len(test)):
                    prod_setosa = train_prob_setosa
                    prod_not_setosa = train_prob_not_setosa
                    

                    for feature in test.columns[:-1]:
                        prod_setosa *= train_probabilities[Labels.SETOSA][feature][test[feature].iloc[row]]
                        prod_not_setosa *= train_probabilities[Labels.NOTSETOSA][feature][test[feature].iloc[row]]

                    if abs(prod_setosa - prod_not_setosa) >= threshold:
                        # print("class: ", test[Labels.CLASS].iloc[row])
                        if test[Labels.CLASS].iloc[row] == Labels.SETOSA:
                            tp += 1
                        else:
                            fp += 1
                # print("tp: ", tp)
                # print("fp: ", fp)

                tpr = tp/true_pos
                fpr = fp/true_neg
                roc[bin_string][0].append(fpr)
                roc[bin_string][1].append(tpr)

        fig = plt.figure(split_num)
        for i, key in enumerate(roc):
            plt.plot(roc[key][0], roc[key][1], label=key)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC of Setosa vs. Not Setosa')
            plt.legend()
        fig.savefig("../figures/roc_curve_plot_" + str(split_num) + ".png")
            # print(roc)

    print(accuracies)
    print(F1_measure)

    fig5 = plt.figure(5)
    for key in accuracies:
        plt.plot(range(constant.NUM_SPLITS), accuracies[key], label=key, linestyle="--")
        plt.xlabel('Round Number')
        plt.ylabel('Accuracy Score')
        plt.xticks(range(constant.NUM_SPLITS))
        plt.title('Plot of Accuracy Scores')
        plt.legend()

        print(key + ": ")
        print("\tAccuracies: " + str(accuracies[key]))   
        print("\tMin Acc: " + str(min(accuracies[key])))
        print("\tMax Acc: " + str(max(accuracies[key])))
        print("\tAvg Acc: " + str(mean(accuracies[key])))

    fig5.savefig("../figures/accuracy_score_plot.png")
    
    fig6 = plt.figure(6)
    # ls = ['-', ':', '-.', '--']
    for i, key in enumerate(F1_measure):
        plt.plot(range(constant.NUM_SPLITS), F1_measure[key], label=key, linestyle="--")
        plt.xlabel('Round Number')
        plt.ylabel('F1 Scores')
        plt.xticks(range(constant.NUM_SPLITS))
        
        plt.title('Plot of F1 Scores')
        plt.legend()
    fig6.savefig("../figures/f1_score_plot.png")

                    

if __name__ == "__main__":
    main()