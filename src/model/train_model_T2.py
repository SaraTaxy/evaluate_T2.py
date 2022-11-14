import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import collections
import yaml
import ssl

import src.utils.util_general as util_general
import src.utils.util_data as util_data
import src.utils.util_model as util_model

torch.cuda.empty_cache()
os.environ['TORCH_HOME'] = './models/cnn/pretrained'
ssl._create_default_https_context = ssl._create_unverified_context

# Configuration file

#vanno commentate su alvis
cfg_file = "./configs/sfcn2/T2/sfcn2_t2_14.yaml"
with open(cfg_file) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

args = util_general.get_args()
args.cfg_file = "./configs/sfcn2/T2/sfcn2_t2_14.yaml"
with open(cfg_file) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

# Seed everything
util_general.seed_all(cfg['seed'])

# Parameters
exp_name = cfg['exp_name']
classes = cfg['data']['classes']
task = "".join(map(str, sorted(classes)))
model_name = cfg['model']['model_name']
cv = cfg['data']['cv']
fold_list = list(range(cv))

# Device
device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(device)

# Files and Directories
fold_dir = os.path.join(cfg['data']['fold_dir'], task)
model_dir = os.path.join(cfg['data']['model_dir'], exp_name)
util_general.create_dir(model_dir)
report_dir = os.path.join(cfg['data']['report_dir'], exp_name)
util_general.create_dir(report_dir)
report_file = os.path.join(report_dir, 'report.xlsx')
plot_training_dir = os.path.join(report_dir, "training")
util_general.create_dir(plot_training_dir)

# CV
results = collections.defaultdict(lambda: [])
acc_cols = []
acc_class_cols = collections.defaultdict(lambda: [])

for fold in fold_list:
    # Dir
    model_fold_dir = os.path.join(model_dir, str(fold))
    util_general.create_dir(model_fold_dir)
    plot_training_fold_dir = os.path.join(plot_training_dir, str(fold))
    util_general.create_dir(plot_training_fold_dir)

    # Results Frame
    acc_cols.append("%s ACC" % str(fold))
    for c in classes[::-1]:
        acc_class_cols[c].append("%s ACC %s" % (str(fold), c))

    # Data Loaders
    fold_data = {step: pd.read_csv(os.path.join(fold_dir, str(fold), '%s.txt' % step), delimiter="\t", index_col='id') for step in ['train', 'val', 'test']}

    #modificato --> ImgDataset_T2
    datasets = {step: util_data.ImgDataset_T2(data=fold_data[step], classes=classes, cfg_data=cfg['data'], step=step) for step in ['train', 'val', 'test']}
    data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                    'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                    'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker)}

    # Model
    print("%s%s%s" % ("*"*50, model_name, "*"*50))
    # util_general.notify_IFTTT("Start %i %s" % (fold, model_name))



    model = util_model.initialize_model(model_name=model_name, num_classes=len(classes), cfg_model=cfg['model'], device=device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Loss function
    weight = [len(fold_data['train']) / (len(classes) * len(fold_data['train']['label'][fold_data['train']['label'] == c])) for c in classes]
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weight)).to(device)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg['trainer']['optimizer']['lr'], weight_decay=cfg['trainer']['optimizer']['weight_decay'])
    # LR Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg['trainer']['scheduler']['mode'], patience=cfg['trainer']['scheduler']['patience'])
    # Train model --> si blocca
    model, history = util_model.train_model(model=model, criterion=criterion, optimizer=optimizer,
                                            scheduler=scheduler, model_name=model_name, data_loaders=data_loaders,
                                            model_dir=model_fold_dir, device=device, cfg_trainer=cfg['trainer'])

    # Plot Training
    util_model.plot_training(history=history, model_name=model_name, plot_training_dir=plot_training_fold_dir)

    # Test model
    test_results = util_model.evaluate(model=model, data_loader=data_loaders['test'], device=device)
    print(test_results)

    # Update report
    results["%s ACC" % str(fold)].append(test_results['all'])
    for c in classes:
        results["%s ACC %s" % (str(fold), str(c))].append(test_results[c])

    # Save Results
    results_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    for c in classes[::-1]:
        results_frame.insert(loc=0, column='std ACC %s' % c, value=results_frame[acc_class_cols[c]].std(axis=1))
        results_frame.insert(loc=0, column='mean ACC %s' % c, value=results_frame[acc_class_cols[c]].mean(axis=1))
    results_frame.insert(loc=0, column='std ACC', value=results_frame[acc_cols].std(axis=1))
    results_frame.insert(loc=0, column='mean ACC', value=results_frame[acc_cols].mean(axis=1))
    results_frame.insert(loc=0, column='model', value=model_name)
    results_frame.to_excel(report_file, index=False)
    # util_general.notify_IFTTT("End %i %s %f" % (fold, model_name, test_results['all']))
