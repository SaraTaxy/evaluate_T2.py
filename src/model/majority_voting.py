import pandas as pd
import numpy as np
import os

directory = os.path.join("./reports/resnet18_14/probability_majority_voting")

acc=[]
for _, _, files in os.walk(directory):
    for file in files:
        if file.endswith('.xlsx'):
            data_T2_T1 = pd.read_excel(directory+"/"+file)
            patient = data_T2_T1.iloc[:, 0].tolist()
            resnet18_0_T1 = data_T2_T1.iloc[:, 1].tolist()
            resnet18_1_T1 = data_T2_T1.iloc[:, 2].tolist()
            resnet18_0_T2 = data_T2_T1.iloc[:, 3].tolist()
            resnet18_1_T2 = data_T2_T1.iloc[:, 4].tolist()
            labels = data_T2_T1.iloc[:, 5].tolist()


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

            for elem in range(0, len(new_label)):
                if new_label[elem] == labels[elem]:
                    corrects += 1
            ac_f = corrects / len(labels)

            acc.append(ac_f)

somma = sum(acc)
acc_tot = somma/(len(acc))

print(acc_tot)

acc.append(acc_tot)

acc = pd.DataFrame(acc)
acc.to_excel('./reports/majority_voting.xlsx')



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