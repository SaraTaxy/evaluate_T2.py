import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
import yaml

import src.utils.util_general as util_general
import src.utils.util_model as util_model

# Configuration file

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



# Seed everything
util_general.seed_all(cfg['seed'])

# Parameters
exp_name = cfg['exp_name']
steps = ["test", "val", "train"]
cv = cfg['data']['cv']
fold_list = list(range(cv))

# Files and Directories
report_dir = cfg['data']['report_dir']
report_dir_exp = os.path.join(report_dir, exp_name)
for elem in cfg['model']:

    prediction_dir = os.path.join(report_dir_exp, elem, "prediction")

    performance_dir = os.path.join(report_dir_exp, elem, "performance")
    util_general.create_dir(performance_dir)

# Load predictions
    results = util_model.get_predictions_FUSION(prediction_dir, fold_list, steps)

# Evaluate
    performance = util_model.get_performance(results, performance_dir)
