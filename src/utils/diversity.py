import pandas as pd
import numpy as np
import os

directory = os.path.join("./reports/resnet18_14/diversity_T1_T2")


fold=[['fold', 'Q_statistic', 'Correlation coeff', 'Disagreement measure', 'Double Fault measure', 'K coef', 'corr matrix']]
for _, _, files in os.walk(directory):
    j = 0
    for file in files:
        if file.endswith('.xlsx'):
            folds = pd.read_excel(directory+"/"+file)
            test = folds.iloc[1:len(folds), 0].tolist()
            fold_T1 = folds.iloc[1:len(folds), 1].tolist()
            fold_T2 = folds.iloc[1:len(folds), 2].tolist()
            fold_true = folds.iloc[1:len(folds), 3].tolist()

            N11 = 0
            N10 = 0
            N01 = 0
            N00 = 0

            for i in range(0, len(fold_T1)):
                if fold_T1[i] == fold_T2[i] and fold_T1[i] == fold_true[i]:
                    N11 += 1
                elif fold_T1[i] != fold_T2[i] and fold_T1[i] == fold_true[i]:
                    N10 += 1
                elif fold_T2[i] != fold_T1[i] and fold_T2[i] == fold_true[i]:
                    N01 += 1
                else:
                    N00 += 1

            total = N11 + N00 + N10 + N01
            correlation_matrix = [[N11, N10], [N01, N00]]
            Q_statistic = ((N11 * N00) - (N01 * N10)) / ((N11 * N00) + (N01 * N10))
            Correlation = ((N11 * N00) - (N01 * N10)) / np.square(((N11 + N10) * (N01 + N00) * (N11 + N01) * (N10 * N00)))
            Disagreement = (N01 + N10) / (total)
            Double_Fault = N00 / total
            K_index = (2*(N11*N00 - N01*N10))/((N11+N10)*(N01+N00)+(N11+N01)*(N10+N00))
            fold_i = [j, Q_statistic, Correlation, Disagreement, Double_Fault, K_index,correlation_matrix]
            j+=1
        fold.append(fold_i)

fold_data = pd.DataFrame(fold)

mean_Q_statistic = np.mean(fold_data.iloc[1:len(folds), 1].tolist())
mean_Correlation = np.mean(fold_data.iloc[1:8, 2].tolist())+np.mean(fold_data.iloc[9:len(folds), 2].tolist())
mean_Disagreement = np.mean(fold_data.iloc[1:len(folds), 3].tolist())
mean_DB = np.mean(fold_data.iloc[1:len(folds), 4].tolist())
mean_K_index = np.mean(fold_data.iloc[1:len(folds), 5].tolist())

a = ['mean', mean_Q_statistic, mean_Correlation, mean_Disagreement, mean_DB, mean_K_index,' ']
fold.append(a)

fold_data = pd.DataFrame(fold)

fold_data.to_excel('./reports/resnet18_14/diversity_T1_T2/diversity_pairwise.xlsx')

