import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import torch
import pandas as pd
import os
import yaml

import src.utils.util_general as util_general
import src.utils.util_data as util_data
import src.utils.util_model as util_model

torch.cuda.empty_cache()

'''
#locale --> per farlo girare sul pc
cfg_file = "./configs/resnet18/resnet18_layer1_2.yaml"
with open(cfg_file) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
'''

#alvis --> non toccare
args = util_general.get_args()
with open(args.cfg_file) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

#Seed everything
util_general.seed_all(cfg['seed'])

# Parameters
exp_name = cfg['exp_name']
classes = cfg['data']['classes']
task = "".join(map(str, sorted(classes)))
model_name = cfg['model_name']
cv = cfg['data']['cv']
fold_list = list(range(cv))

# Device
device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(device)

# Files and Directories
fold_dir_1 = os.path.join(cfg['data']['fold_dir_T1'], task)
fold_dir_2 = os.path.join(cfg['data']['fold_dir_T2'], task)

model_dir = cfg['data']['model_dir']
model_dir_exp = os.path.join(model_dir, exp_name)
report_dir = os.path.join(cfg['data']['report_dir'], exp_name)
prediction_dir = os.path.join(report_dir)
util_general.create_dir(prediction_dir)


print("%s%s%s" % ("*"*50, model_name, "*"*50))
# Predict CV
for fold in fold_list:
    for elem in cfg['model']:
        #break
        # Dir & file
        prediction_dir = os.path.join(report_dir, elem, "prediction")
        util_general.create_dir(prediction_dir)
        model_fold_dir = os.path.join(model_dir_exp, elem, str(fold))
        util_general.create_dir(model_fold_dir)
        prediction_fold_dir = os.path.join(prediction_dir, str(fold))
        util_general.create_dir(prediction_fold_dir)

    # Data Loaders
        fold_data_1 = {step: pd.read_csv(os.path.join(fold_dir_1, str(fold), '%s.txt' % step), delimiter="\t", index_col='id') for step in ['train', 'val', 'test']}
        fold_data_2 = {step: pd.read_csv(os.path.join(fold_dir_2, str(fold), '%s.txt' % step), delimiter="\t", index_col='id') for step in ['train', 'val', 'test']}

        datasets = {step: util_data.ImgDataset_Fusion(data_T1=fold_data_1[step], data_T2=fold_data_2[step], classes=classes,cfg_data=cfg['data'], step=step) for step in ['train', 'val', 'test']}

        data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['data']['batch_size'], shuffle=True,num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['data']['batch_size'], shuffle=False,num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['data']['batch_size'], shuffle=False,num_workers=num_workers, worker_init_fn=util_data.seed_worker)}

        # Split
        for step in ['train', 'val', 'test']:
            prediction_file = os.path.join(prediction_fold_dir, "prediction_%s_%s.xlsx" % (step, fold))
            probability_file = os.path.join(prediction_fold_dir, "probability_%s_%s.xlsx" % (step, fold))

            # Predict Models
            prediction_frame = pd.DataFrame()
            probs_frame = pd.DataFrame()
            # Model
            print("%s%s%s" % ("_" * 50, elem, "_" * 50))
            model = torch.load(os.path.join(model_fold_dir, "%s.pt" % model_name), map_location=device)
            model = model.to(device)

            # Prediction
            predict = util_model.get_predict_function_fusion(output_type=cfg['model'].get('output_type', 'single'))
            predictions, probabilities, truth = predict(model, data_loaders[step], device)

            # Update report
            prediction_frame[model_name] = pd.Series(predictions)
            for i in range(len(classes)):
                probs_frame["%s_%i_T1" % (model_name, i)] = pd.Series({x: probs[i] for x, probs in probabilities.items()})

            # Ground Truth
            prediction_frame["True"] = pd.Series(truth)
            probs_frame["True"] = pd.Series(truth)

            # Save Results
            prediction_frame.to_excel(prediction_file, index=True)
            probs_frame.to_excel(probability_file, index=True)


