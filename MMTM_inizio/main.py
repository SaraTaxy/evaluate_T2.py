from Utils import*

'''
tipofunzione = 'train'
fold = 0
continue_learning = 'false'
model_type = 'resnet34'
'''

tipofunzione = sys.argv[1]
fold = int(sys.argv[2])
continue_learning = sys.argv[3]
model_type = sys.argv[4]
cuda_id = sys.argv[5]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=cuda_id

cuda_str = 'cuda:' + cuda_id

print(cuda_str)
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")
print(device)


tipoNormalizzazione_str = 'None'
loss = 'cross'
print('tipofunzione ' + tipofunzione + ' - ' + 'fold ' + str(fold) +  ' - ' + 'continue_learning ' + continue_learning + ' tipoNormalizzazione_str ' + tipoNormalizzazione_str)
  
if continue_learning == 'true':
  continue_learning = True
else:
  continue_learning = False
  
esclusi = ['115-116']
dfs = pd.read_excel('D:/Michi_Campus_Biomedico/NewFold_10_Confronto.xlsx', sheet_name='Foglio1')
classes = ['0_cavo', '1_cavo']
basePath = 'D:/Michi_Campus_Biomedico/DatasetDCE_Water_DWI_clinico'

pathforDCEWeights = 'DCE_3DAll_BoxIsotropic_lr1_cross_'+ model_type
pathforWATERWeights = 'Water_3DAll_BoxIsotropic_lr1_cross_'+ model_type
pathforDWIWeights = 'DWI_3DAll_BoxIsotropic_lr1_cross_'+ model_type
filepesi = 'best_model_weights.pth'

minDimLesion = 30
fattore = 0.1
epocheFineTuning = 200
learningRate = 0.00001
weightDecay = 0.0001
batchSize = 16
num_epoch = 300
ch = 8
'''
minDimLesion = 30
fattore = 0.1
epocheFineTuning = 3
learningRate = 0.00001
weightDecay = 0.0001
batchSize = 16
num_epoch = 3
ch = 8
'''

test_set = dfs[dfs.FOLD == fold].ID.to_list()
valFold = dfs[dfs.FOLD == fold].Fold_to_use.values[0]
vali_set = dfs[dfs.FOLD == valFold].ID.to_list()
print('Test: fold ' + str(fold))
print(test_set)
print('Val: fold ' + str(valFold))
print(vali_set)
  
if len(list(set(test_set) & set(vali_set)))>0:
  print('errore!!!')
  
outputPath = 'D:/Michi_Campus_Biomedico/Results/D_W_D_MT_clinic_' + model_type+ '/Fold' + str(fold) +'/'

pathforDCEWeights = 'D:/Michi_Campus_Biomedico/' + pathforDCEWeights + '/Fold' + str(fold) +'/' + filepesi
pathforWATERWeights = 'D:/Michi_Campus_Biomedico/' + pathforWATERWeights + '/Fold' + str(fold) +'/' + filepesi
pathforDWIWeights = 'D:/Michi_Campus_Biomedico/' + pathforDWIWeights + '/Fold' + str(fold) +'/' + filepesi

try:
  os.makedirs(outputPath)
except OSError:
  pass
    
with open(outputPath + 'InfoAddestramento.txt', "w") as file_object:
  file_object.write('batch_size: '+str(batchSize)+'\n')
  file_object.write('lr: '+str(learningRate)+'\n')
  file_object.write('decay: '+str(weightDecay)+'\n')
  file_object.write('minDimLesion: '+ str(minDimLesion)+'\n')
  file_object.write('fattore per il fine tuning sul val: '+ str(fattore)+'\n')
  file_object.write('epocheFineTuning: '+ str(epocheFineTuning)+'\n')
  file_object.write('tipoNormalizzazione: '+ tipoNormalizzazione_str +'\n')
  file_object.write('Range 01')
    
  
if tipofunzione == 'train':
  main_TRAIN(fold,continue_learning, tipoNormalizzazione_str, loss,
             basePath, classes, ch, learningRate, weightDecay, batchSize, num_epoch, 
             vali_set, test_set, esclusi, minDimLesion,
             outputPath, pathforDCEWeights, pathforWATERWeights, pathforDWIWeights, model_type)
  
  main_validation(fold,False, tipoNormalizzazione_str, loss,
                    basePath, classes, ch, learningRate, weightDecay, batchSize, epocheFineTuning, fattore, 
                    vali_set, test_set, esclusi, minDimLesion,
                    outputPath, pathforDCEWeights, pathforWATERWeights, pathforDWIWeights, model_type)
  
  main_final_restrain(fold,False, tipoNormalizzazione_str, loss,
                     basePath, classes, ch, learningRate, weightDecay, batchSize, 
                     vali_set, test_set, esclusi, minDimLesion,
                     outputPath, pathforDCEWeights, pathforWATERWeights, pathforDWIWeights, model_type)
  
elif tipofunzione == 'val':
  main_validation(fold,continue_learning, tipoNormalizzazione_str,  loss,
                  basePath, classes, ch, learningRate, weightDecay, batchSize, epocheFineTuning, fattore,
                  vali_set, test_set, esclusi, minDimLesion,
                  outputPath, pathforDCEWeights, pathforWATERWeights, pathforDWIWeights, model_type)
  
  main_final_restrain(fold,False, tipoNormalizzazione_str, loss,
                     basePath, classes, ch, learningRate, weightDecay, batchSize, 
                     vali_set, test_set, esclusi, minDimLesion,
                     outputPath, pathforDCEWeights, pathforWATERWeights, pathforDWIWeights, model_type)
  
else:
  print('RETRAIN SELEZIONATO')
  main_final_restrain(fold,continue_learning, tipoNormalizzazione_str,  loss,
                     basePath, classes, ch, learningRate, weightDecay, batchSize, 
                     vali_set, test_set, esclusi, minDimLesion,
                     outputPath, pathforDCEWeights, pathforWATERWeights, pathforDWIWeights, model_type) 