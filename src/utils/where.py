import sys;
import re
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
import torch
import yaml
import ssl

torch.cuda.empty_cache()
os.environ['TORCH_HOME'] = './models/cnn/pretrained'
ssl._create_default_https_context = ssl._create_unverified_context

# Configuration file

#locale --> per farlo girare sul pc
cfg_file = "./configs/MMTM_layer_fusion.yaml"
with open(cfg_file) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

acc_max_0 = cfg['fusion_0']['acc_fusion_0']

fusion_1 = list(cfg['fusion_1'].items())
fusion_2_1 = list(cfg['fusion_2_1'].items())
fusion_2_2 = list(cfg['fusion_2_2'].items())
fusion_2_3 = list(cfg['fusion_2_3'].items())
fusion_2_4 = list(cfg['fusion_2_4'].items())
fusion_3_1 = list(cfg['fusion_3_1'].items())
fusion_3_2 = list(cfg['fusion_3_2'].items())
fusion_3_3 = list(cfg['fusion_3_3'].items())
fusion_3_4 = list(cfg['fusion_3_4'].items())
fusion_4_1 = list(cfg['fusion_4_1'].items())

maxi = 0

for elem in range(0, len(fusion_1)):
    if fusion_1[elem][1] > maxi:
        maxi = fusion_1[elem][1]
        index = fusion_1[elem][0]

print(index, maxi)

layer_1_i = re.findall(r'\d+', index)
layer_1_i = [int(elem) for elem in layer_1_i]

print(layer_1_i)
index = []
for i in layer_1_i:
    if i == 1:
        maxi_1_1 = 0
        for elem in range(0, len(fusion_2_1)):
            if fusion_2_1[elem][1] > maxi_1_1:
                maxi_1_1 = fusion_2_1[elem][1]
                index_1_1 = fusion_2_1[elem][0]
        print(index_1_1, maxi_1_1)
        index.append(index_1_1)
        #print("qui")
    if i == 2:
        maxi_2_1 = 0
        for elem in range(0, len(fusion_2_2)):
            if fusion_2_2[elem][1] > maxi_2_1:
                maxi_2_1 = fusion_2_2[elem][1]
                index_2_1 = fusion_2_2[elem][0]
        print(index_2_1, maxi_2_1)
        index.append(index_2_1)
        #print("qu0")
    if i == 3:
        maxi_3_1 = 0
        for elem in range(0, len(fusion_2_3)):
            if fusion_2_3[elem][1] > maxi_3_1:
                maxi_3_1 = fusion_2_3[elem][1]
                index_3_1 = fusion_2_3[elem][0]
        print(index_3_1, maxi_3_1)
        index.append(index_3_1)
        #print("qua")
    if i == 4:
        maxi_4_1 = 0
        for elem in range(0, len(fusion_2_4)):
            if fusion_2_4[elem][1] > maxi_4_1:
                maxi_4_1 = fusion_2_4[elem][1]
                index_4_1 = fusion_2_4[elem][0]
        print(index_4_1, maxi_4_1)
        index.append(index_4_1)
        #print("quq")


for elem in index:
    layer_2_i = re.findall(r'\d+', elem)
    layer_2_i = [int(elem) for elem in layer_2_i]

layer_2 = layer_2_i[1]

if layer_2 == 1:
    maxi_1 = 0
    for elem in range(0, len(fusion_3_1)):
        if fusion_3_1[elem][1] > maxi_1:
            maxi_1 = fusion_3_1[elem][1]
            index_1 = fusion_3_1[elem][0]

    print(index_1, maxi_1)

    if cfg['fusion_4_1']['acc_fusion_layer_1_2_3_4'] > maxi_1:
        structur_final = cfg['fusion_4_1'].items()
    else:
        structur_final = index_1

if layer_2 == 2:
    maxi_2 = 0
    for elem in range(0, len(fusion_3_2)):
        if fusion_3_2[elem][1] > maxi_2:
            maxi_2 = fusion_3_2[elem][1]
            index_2 = fusion_3_2[elem][0]

    print(index_2, maxi_2)

    if cfg['fusion_4_1']['acc_fusion_layer_1_2_3_4'] > maxi_2:
        structur_final = cfg['fusion_4_1'].items()
    else:
        structur_final = index_2

if layer_2 == 3:
    maxi_3 = 0
    for elem in range(0, len(fusion_3_3)):
        if fusion_3_3[elem][1] > maxi_3:
            maxi_3 = fusion_3_3[elem][1]
            index_3 = fusion_3_3[elem][0]

    print(index_3, maxi_3)

    if cfg['fusion_4_1']['acc_fusion_layer_1_2_3_4'] > maxi_3:
        structur_final = cfg['fusion_4_1'].items()
    else:
        structur_final = index_3

if layer_2 == 4:
    maxi_4 = 0
    for elem in range(0, len(fusion_3_4)):
        if fusion_3_4[elem][1] > maxi_4:
            maxi_4 = fusion_3_4[elem][1]
            index_4 = fusion_3_4[elem][0]

    print(index_4, maxi_4)

    if cfg['fusion_4_1']['acc_fusion_layer_1_2_3_4'] > maxi_4:
        structur_final = cfg['fusion_4_1'].items()
    else:
        structur_final = index_4



print(structur_final)