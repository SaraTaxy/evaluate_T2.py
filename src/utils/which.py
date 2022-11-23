import pandas as pd
import numpy as np


data_T2_T1 = pd.read_excel('./reports/T1_T2_metriche.xlsx')
print(data_T2_T1)

mean_acc_T2 = data_T2_T1.iloc[:, 1].tolist()
mean_acc_T1 = data_T2_T1.iloc[:, 2].tolist()

f_score_T2= data_T2_T1.iloc[:, 3].tolist()
f_score_T1=data_T2_T1.iloc[:, 4].tolist()

network_name = data_T2_T1.iloc[:, 0].tolist()

max_acc = 0
sum_acc = 0
acc_T1_max = 0
acc_T2_max = 0

max_f_score = 0
sum_f_score = 0
f_score_T1_max = 0
f_score_T2_max = 0

s = []
f = []


if len(network_name) == len(mean_acc_T2) and len(network_name) == len(mean_acc_T1):
    for elem in range(0, (len(network_name))):
        sum_acc = mean_acc_T1[elem] + mean_acc_T2[elem]
        s.append(sum_acc)
        sum_f_score = f_score_T1[elem]+f_score_T2[elem]
        f.append(sum_f_score)
        if sum_acc > max_acc:
            max_acc = sum_acc
            acc_T2_max = mean_acc_T2[elem]
            acc_T1_max = mean_acc_T1[elem]
            index_max_acc = network_name[elem]
        if sum_f_score > max_f_score:
            max_f_score = sum_f_score
            f_score_T2_max = f_score_T2[elem]
            f_score_T1_max = f_score_T1[elem]
            index_max_f_score = network_name[elem]


print("Network migliore per T1 e T2 acc : ", index_max_acc)
print("valore di max acc T1: ", acc_T1_max)
print("valore di max acc T2 : ", acc_T2_max)

print("Network migliore per T1 e T2 f_score : ", index_max_f_score)
print("valore di max f_score T1: ", f_score_T1_max)
print("valore di max af_score T2 : ", f_score_T2_max)


#print(max_acc)
#max_1 = np.max(s)   #check
s_d = pd.DataFrame(s)
f_d = pd.DataFrame(f)

data_T2_T1['sum ACC'] = s_d
data_T2_T1['sum F score'] = f_d

data_T2_T1.loc[len(data_T2_T1.index)] = [" ", " ", " ", " ", " ", " ", " "]
data_T2_T1.loc[len(data_T2_T1.index)] = [" ", " ", " ", " ", " ", " ", " "]

data_T2_T1.loc["summary acc "] = [index_max_acc, acc_T1_max, acc_T2_max, max_acc, " ", " ", " "]

data_T2_T1.loc[len(data_T2_T1.index)] = [" ", " ", " ", " ", " ", " ", " "]
data_T2_T1.loc[len(data_T2_T1.index)] = [" ", " ", " ", " ", " ", " ", " "]

data_T2_T1.loc["summary f_score "] = [index_max_f_score, f_score_T1_max, f_score_T2_max, max_f_score, " ", " ", " "]

data_T2_T1.to_excel('./reports/report_T1_T2_immagini.xlsx')







