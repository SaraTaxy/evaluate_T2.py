import pandas as pd
import numpy as np
import os

directory = os.path.join("./reports/resnet18_14/probability_weighted_voting")

acc = ['accuracy']
precision = ['precision']
recall = ['recall']
specificity = ['specificity']
f_score= ['F_score']
for _, _, files in os.walk(directory):
    for file in files:
        if file.endswith('.xlsx'):
            data_T2_T1 = pd.read_excel(directory+"/"+file)
            patient = data_T2_T1.iloc[:, 0].tolist()
            resnet18_0_T1 = data_T2_T1.iloc[2:len(patient), 1].tolist()
            resnet18_1_T1 = data_T2_T1.iloc[2:len(patient), 2].tolist()
            resnet18_0_T2 = data_T2_T1.iloc[2:len(patient), 3].tolist()
            resnet18_1_T2 = data_T2_T1.iloc[2:len(patient), 4].tolist()
            labels = data_T2_T1.iloc[2:len(patient), 5].tolist()

            new_label = []
            for i in range(0, len(resnet18_0_T2)):
                sum_0 = (resnet18_0_T1[i] + resnet18_0_T2[i])
                mean_0 = sum_0 / 2

                sum_1 = (resnet18_1_T1[i] + resnet18_1_T2[i])
                mean_1 = sum_1 / 2

                if sum_0 > sum_1:
                    n_l = 0
                else:
                    n_l = 1

                new_label.append(n_l)

            corrects = 0
            TP = 0
            FP = 0
            FN = 0
            TN = 0

            for elem in range(0, len(new_label)):
                if new_label[elem] == labels[elem]:
                    corrects += 1
                    if new_label[elem] == 1:
                        TP += 1
                    else:
                        TN += 1
                if new_label[elem] != labels[elem] and labels[elem] == 1:
                    FP +=1
                if new_label[elem] != labels[elem] and labels[elem] == 0:
                    FN +=1


            ac_f = corrects / len(labels)
            precision_f = TP/(TP+FP)
            recall_f = TP/(TP+FN)
            specificity_f = TN/(FP+TN)
            f_score_f = 2 * ((precision_f*recall_f)/(precision_f+recall_f))

            acc.append(ac_f)
            precision.append(precision_f)
            recall.append(recall_f)
            specificity.append(specificity_f)
            f_score.append(f_score_f)

acc_mean = sum(acc[1:len(acc)])/(len(acc)-1)

precision_mean = sum(precision[1:len(precision)])/(len(precision)-1)

recall_mean= sum(recall[1:len(recall)])/(len(recall)-1)

specificity_mean= sum(specificity[1:len(specificity)])/(len(specificity)-1)

f_score_mean= sum(f_score[1:len(f_score)])/(len(f_score)-1)

print(acc_mean)
print(precision_mean)
print(recall_mean)
print(specificity_mean)
print(f_score_mean)

acc.append(acc_mean)
precision.append(precision_mean)
recall.append(recall_mean)
specificity.append(specificity_mean)
f_score.append(f_score_mean)


tot = [acc, precision, recall, specificity, f_score]

tot = pd.DataFrame(tot).transpose()
tot.to_excel(directory+'/'+"weighted_voting.xlsx")



#------------------------------------------------------un solo file


data_T2_T1 = pd.read_excel('./reports/resnet18_14/probability_majority_voting/probablilty_T1_T2_fold0.xlsx')
print(data_T2_T1)

patient = data_T2_T1.iloc[:, 0].tolist()
resnet18_0_T1 = data_T2_T1.iloc[:, 1].tolist()
resnet18_1_T1 = data_T2_T1.iloc[:, 2].tolist()
resnet18_0_T2 = data_T2_T1.iloc[:, 3].tolist()
resnet18_1_T2 = data_T2_T1.iloc[:, 4].tolist()
labels = data_T2_T1.iloc[:,5].tolist()


new_label = []
for i in range(0, len(resnet18_0_T2)):
    sum_0 = (resnet18_0_T1[i]+resnet18_0_T2[i])
    mean_0 = sum_0/2

    sum_1 = (resnet18_1_T1[i] + resnet18_1_T2[i])
    mean_1 = sum_1 / 2

    if sum_0 > sum_1 :
        n_l = 0
    else:
        n_l = 1

    new_label.append(n_l)

print(new_label)
print(labels)

corrects = 0
for elem in range(0, len(new_label)):
    if new_label[elem] == labels[elem]:
        corrects += 1

acc = corrects/len(labels)
print(acc)