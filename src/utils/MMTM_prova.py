import torch

import torch.nn as nn

import src.utils.util_general as util_general
import src.utils.sfcn as sfcn
import src.utils.resnet as resnet


import src.utils.util_general as util_general
import src.utils.util_data as util_data
import src.utils.util_model as util_model



#provato con due due input uguali x = input.float() e funziona

class MMTM(nn.Module):
    def __init__(self, channel_T1, channel_T2):
        super(MMTM, self).__init__()

        # da B a Sb   da A a Sa
        self.avg1 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avg2 = nn.AdaptiveAvgPool3d((1, 1, 1))

        ratio = int((channel_T1 + channel_T2) / 4)  # Cz  --> ha 3 input

        self.fc_1 = nn.Sequential(                    #serve per la concatenazione
            nn.Linear((channel_T1+channel_T2), ratio),
            nn.ReLU(inplace=True)
        )

        self.vector_T1 = nn.Sequential(
            nn.Linear(ratio, channel_T1),
            nn.Sigmoid()
        )

        self.vector_T2 = nn.Sequential(
            nn.Linear(ratio, channel_T2),
            nn.Sigmoid()
        )

    def forward(self, T1, T2):   #dove gli do in input le immagini?

        # Sa e Sb
        T1_av = torch.flatten(self.avg1(T1), 1)  # vettore T1 --> sarevìbbe x = input.float()
        T2_av = torch.flatten(self.avg2(T2), 1)  # vettore T2

        # Z  --> concatenazione
        fusion = self.fc_1(torch.cat((T1_av, T2_av), 1))

        T1_Ea = self.vector_T1(fusion)     #T1
        T2_Eb = self.vector_T2(fusion)     #T2

        # channel wise operation  --> #B .x Eb = B_tilde (2--> foglio 1)

        T1_f = 2 * T1 * T1_Ea.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        T2_f = 2 * T2 * T2_Eb.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        #T1_f = T1 * T1_Ea.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        #T2_f = T2 * T2_Eb.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        return T1_f, T2_f



class FusionNetwork(nn.Module):
    def __init__(self, network_name):
        super().__init__()

        self.typeNet_name = network_name

        #print("** ", network_name, " **" )
        self.MMTM1 = MMTM(64,64)
        self.MMTM2 = MMTM(128,128)
        self.MMTM3 = MMTM(256,256)
        self.MMTM4 = MMTM(512, 512)

    def forward(self, x1, x2, switch = [True, False, False, False]):

        #todo: controlla le dimensioni dei canali e capisci anche model come funziona

        x1 = self.typeNet_name.conv1(x1)
        x2 = self.typeNet_name.conv1(x2)
        x1 = self.typeNet_name.bn1(x1)
        x2 = self.typeNet_name.bn1(x2)
        x1 = self.typeNet_name.relu(x1)
        x2 = self.typeNet_name.relu(x2)
        x1 = self.typeNet_name.maxpool(x1)
        x2 = self.typeNet_name.maxpool(x2)
        x1 = self.typeNet_name.layer1(x1)
        x2 = self.typeNet_name.layer1(x2)

        if switch[0]:
            x1,x2 = self.MMTM1(x1, x2)

        x1 = self.typeNet_name.layer2(x1)
        x2 = self.typeNet_name.layer2(x2)

        if switch[1]:
            x1,x2 = self.MMTM2(x1,x2)

        x1 = self.typeNet_name.layer3(x1)
        x2 = self.typeNet_name.layer3(x2)

        if switch[2]:
            x1,x2 = self.MMTM3(x1, x2)

        x1 = self.typeNet_name.layer4(x1)
        x2 = self.typeNet_name.layer4(x2)

        if switch[3]:
            x1,x2 = self.MMTM4(x1, x2)

        x_f_1 = self.typeNet_name.avgpool(x1)
        x_f_2 = self.typeNet_name.avgpool(x2)

        x_f_1 = x_f_1.view(x_f_1.size(0), -1)
        x_f_2 = x_f_2.view(x_f_2.size(0), -1)

        x_f_1 = self.typeNet_name.fc(x_f_1)
        x_f_2 = self.typeNet_name.fc(x_f_2)

        return x_f_1, x_f_2





