import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean


def count(data, colname, bin_num, target):
    condition = (data[colname] == bin_num) & (data['class'] == target)
    return len(data[condition])

def main() -> None:
    NUM_SPLITS = 5
    TRAIN_SIZE = 2/3
    BIN_SIZES = [5, 10, 15, 20]
    CLASS = 'class'
    SETOSA = 'Iris-setosa'
    NOTSETOSA = 'Iris-versicolor'

    data = pd.read_csv("../data/iris.csv")
    data[CLASS] = data[CLASS].replace('Iris-virginica', 'Iris-versicolor')
    accuracies = {bin_s: [] for bin_s in BIN_SIZES}
    F1_measure = {bin_s: [] for bin_s in BIN_SIZES}
    FPR = {bin_s: [] for bin_s in BIN_SIZES}
    TPR = {bin_s: [] for bin_s in BIN_SIZES}
    # print(data)
    
    for split_num in range(NUM_SPLITS):
        data = data.sample(frac=1) # shuffle the data

        roc = {5: [[],[]], 10: [[],[]], 15: [[],[]], 20: [[],[]]}
        roc = {bin_s: [[],[]] for bin_s in BIN_SIZES}
        for bin_sz in BIN_SIZES:
            data_copy = data.copy(deep=True)

            # discretize the data
            for j in data_copy.columns[:-1]:
                avg = mean(data_copy[j])
                data_copy[j] = data_copy[j].replace(0, avg)
                data_copy[j] = pd.cut(data_copy[j], bins=bin_sz, labels=False)

            # split into test and train
            train_len = int(TRAIN_SIZE*len(data_copy))
            train = data_copy.iloc[:train_len,:]
            test = data_copy.iloc[train_len:,:]
            
            # calculate train probabilities
            n_yes = count(train, CLASS, SETOSA, SETOSA)
            n_no = count(train, CLASS, NOTSETOSA, NOTSETOSA)
            train_probabilities = {SETOSA:{}, NOTSETOSA:{}}

            for column in train.columns[:-1]:
                train_probabilities[SETOSA][column] = {}
                train_probabilities[NOTSETOSA][column] = {}

                for i in range(bin_sz):
                    nc_yes = count(train, column, i, SETOSA)
                    nc_no = count(train, column, i, NOTSETOSA)

                    train_probabilities[SETOSA][column][i] = (nc_yes + 1)/(n_yes + bin_sz)
                    train_probabilities[NOTSETOSA][column][i] = (nc_no+ 1)/(n_no + bin_sz)
            
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
                    prod_setosa *= train_probabilities[SETOSA][feature][test[feature].iloc[row]]
                    prod_not_setosa *= train_probabilities[NOTSETOSA][feature][test[feature].iloc[row]]
            
                min_diff = min(abs(prod_setosa - prod_not_setosa), min_diff)
                max_diff = max(abs(prod_setosa - prod_not_setosa), max_diff)

                if prod_setosa > prod_not_setosa:
                    if test[CLASS].iloc[row] == SETOSA:
                        true_pos += 1
                    else:
                        false_pos += 1
                else:
                    if test[CLASS].iloc[row] == NOTSETOSA:
                        true_neg += 1
                    else:
                        false_neg += 1
            

            accuracies[bin_sz].append((true_pos + true_neg) / len(test))

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
                for row in range(len(test)):
                    prod_setosa = train_prob_setosa
                    prod_not_setosa = train_prob_not_setosa
                    
                    for feature in test.columns[:-1]:
                        prod_setosa *= train_probabilities[SETOSA][feature][test[feature].iloc[row]]
                        prod_not_setosa *= train_probabilities[NOTSETOSA][feature][test[feature].iloc[row]]

                    if abs(prod_setosa - prod_not_setosa) >= threshold:
                        if test[CLASS].iloc[row] == SETOSA:
                            tp += 1
                        else:
                            fp += 1

                tpr = tp/true_pos
                fpr = fp/true_neg
                roc[bin_sz][0].append(fpr)
                roc[bin_sz][1].append(tpr)

        fig = plt.figure(split_num)
        for i, key in enumerate(roc):
            plt.plot(roc[key][0], roc[key][1], label=key, linestyle='--', dashes=(5, i))
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC of Setosa vs. Not Setosa')
            plt.legend()
        fig.savefig("../figures/roc_curve_plot_" + str(split_num) + ".png")



    fig5 = plt.figure(5)
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

    fig5.savefig("../figures/accuracy_score_plot.png")
    
    fig6 = plt.figure(6)
    for i, key in enumerate(F1_measure):
        plt.plot(range(NUM_SPLITS), F1_measure[key], label=str(key) + " bins:", linestyle='--', dashes=(5, i))
        plt.xlabel('Round Number')
        plt.ylabel('F1 Scores')
        plt.xticks(range(NUM_SPLITS))
        
        plt.title('Plot of F1 Scores')
        plt.legend()
    fig6.savefig("../figures/f1_score_plot.png")

                    

if __name__ == "__main__":
    main()