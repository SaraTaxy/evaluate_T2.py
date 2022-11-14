import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
import pandas as pd
import yaml

import src.utils.util_general as util_general

# Configuration file
args = util_general.get_args()
args.cfg_file = "./configs/sfcn2/T2/sfcn2_t2_14.yaml"
with open(cfg_file) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
'''
cfg_file = "./configs/sfcn2/T2/sfcn2_t2_14.yaml"
with open(cfg_file) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
'''

# Seed Everything
util_general.seed_all(cfg['seed'])

# Params
exp_name = cfg['exp_name']
classes = cfg['data']['classes']
task = "".join(map(str, sorted(classes)))
y_label = "Final"
cv = cfg['data']['cv']

# Files and Directories
fold_file = os.path.join("./data/processed", "class_folds")
data_dir = "./data/processed"
fold_dir = os.path.join(data_dir, "folds", task)
util_general.create_dir(fold_dir)

# Load data
fold_data = pd.read_csv(fold_file, dtype={'RecordID': str})
fold_data = fold_data.set_index('RecordID')

# Select Classes
fold_data = fold_data[fold_data[y_label].isin(classes)]

# all.txt
with open(os.path.join(fold_dir, 'all.txt'), 'w') as file:
    file.write("id\timg\tlabel\n")
    for patient in fold_data.index:
        label = fold_data.loc[patient, y_label]
        row = "%s\tRecordID_%s_T2_flair.nii\t%s\n" % (patient, patient, label)
        file.write(row)

# create split dir
steps = ['train', 'val', 'test']
for fold in range(cv):
    dest_dir_cv = os.path.join(fold_dir, str(fold))
    util_general.create_dir(dest_dir_cv)

    # .txt
    for step in steps:
        fold_data_step = fold_data[fold_data[str(fold)] == step]
        with open(os.path.join(dest_dir_cv, '%s.txt' % step), 'w') as file:
            file.write("id\timg\tlabel\n")
            for patient in fold_data_step.index:
                label = fold_data_step.loc[patient, y_label].item()
                row = "%s\tRecordID_%s_T2_flair.nii\t%s\n" % (patient, patient, label)
                file.write(row)
