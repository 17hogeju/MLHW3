from matplotlib.lines import lineStyles
import pandas as pd
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
    
    
    for i in range(constant.NUM_SPLITS):
        data = data.sample(frac=1) # shuffle the data
        data_copy = data.copy(deep=True) # not really sure if this line is necessary
        for bin_sz in constant.BIN_SIZES:
            data_copy = data.copy(deep=True)

            # discretize the data
            for j in data_copy.columns[:-1]:
                avg = mean(data_copy[j])
                data_copy[j] = data_copy[j].replace(0, avg)
                data_copy[j] = pd.cut(data_copy[j], bins=bin_sz, labels=False)

            # split into test and train
            train_len = int(constant.TRAIN_SIZE*len(data_copy))
            train_x = data_copy.iloc[:train_len,:]
            test_x = data_copy.iloc[train_len:,:]

            count_setosa = count(train_x, Labels.CLASS, Labels.SETOSA, Labels.SETOSA)
            count_not_setosa = count(train_x, Labels.CLASS, Labels.NOTSETOSA, Labels.NOTSETOSA)

            prob_setosa = count_setosa/len(train_x)
            prob_not_setosa = count_not_setosa/len(train_x)

            probabilities = {Labels.SETOSA:{}, Labels.NOTSETOSA:{}}
            
            # caluclate probabilities
            for column in train_x.columns[:-1]:
                probabilities[Labels.SETOSA][column] = {}
                probabilities[Labels.NOTSETOSA][column] = {}

                for i in range(bin_sz):
                    count_bin_setosa = count(train_x, column, i, Labels.SETOSA)
                    count_bin_not_setosa = count(train_x, column, i, Labels.NOTSETOSA)

                    probabilities[Labels.SETOSA][column][i] = count_bin_setosa /count_setosa
                    probabilities[Labels.NOTSETOSA][column][i] = count_bin_not_setosa / count_not_setosa

            true_pos,true_neg, false_pos,false_neg = 0,0,0,0

            for row in range(len(test_x)):
                prod_setosa = prob_setosa
                prod_not_setosa = prob_not_setosa

                for feature in test_x.columns[:-1]:
                    prod_setosa *= probabilities[Labels.SETOSA][feature][test_x[feature].iloc[row]]
                    prod_not_setosa *= probabilities[Labels.NOTSETOSA][feature][test_x[feature].iloc[row]]
                    
                
                if prod_setosa > prod_not_setosa:
                    if test_x[Labels.CLASS].iloc[row] == Labels.SETOSA:
                        true_pos += 1
                    else:
                        false_pos += 1
                else:
                    if test_x[Labels.CLASS].iloc[row] == Labels.NOTSETOSA:
                        true_neg += 1
                    else:
                        false_neg += 1

            bin_string = "bin_"+str(bin_sz)
            accuracies[bin_string].append((true_pos+true_neg) / len(test_x))

            # Calculate the F1 scores
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)

            F1_measure[bin_string].append(2*precision*recall/(precision+recall))
            TPR[bin_string].append(true_pos / (true_pos + false_neg))
            FPR[bin_string].append(false_pos/ (true_neg+false_pos))

    
    
    fig1 = plt.figure()
    for key in accuracies:
        plt.plot(range(constant.NUM_SPLITS), accuracies[key], label=key, alpha=0.6)
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

    fig1.savefig("../figures/accuracy_score_plot.png")
    
    fig2 = plt.figure()
    # ls = ['-', ':', '-.', '--']
    for i, key in enumerate(F1_measure):
        plt.plot(range(constant.NUM_SPLITS), F1_measure[key], label=key, linestyle="--", dashes=(5, pow(2, i)))
        plt.xlabel('Round Number')
        plt.ylabel('F1 Scores')
        plt.xticks(range(constant.NUM_SPLITS))
        
        plt.title('Plot of F1 Scores')
        plt.legend()
    fig2.savefig("../figures/f1_score_plot.png")

    fig3 = plt.figure()
    for i, key in enumerate(FPR):
        plt.plot(FPR[key], TPR[key], label=key, linestyle="--", dashes=(5, pow(2, i)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.xticks(range(constant.NUM_SPLITS))
        plt.title('ROC of Setosa vs. Not Setosa')
        plt.legend()
    fig3.savefig("../figures/roc_curve_plot.png")



    print("F1_scores: ", F1_measure)
    print("FPR: ", FPR)
    print("TPR: ", TPR)

                    

if __name__ == "__main__":
    main()