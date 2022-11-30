import torch
import torch.nn as nn
import time
import copy
import pandas as pd
from tqdm import tqdm
import os
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
import seaborn as sns
import monai

import src.utils.util_general as util_general
import src.utils.sfcn as sfcn
import src.utils.resnet as resnet
#import src.utils.resnetsfcn as resnetsfcn
import src.utils.MMTM_prova as fusion


def freeze_layer_parameters(model, freeze_layers):
    for name, param in model.named_parameters():
        if name.startswith(tuple(freeze_layers)):
            param.requires_grad = False


def initialize_model(model_name, num_classes, cfg_model, device, state_dict=True):
    if model_name == "SFCN":
        model = sfcn.SFCN()
        #model = sfcn.SFCN(avg_shape_T2=cfg_model["avg_shape_T2"])
        model = torch.nn.DataParallel(model)
        if cfg_model["pretrained"]:
            if state_dict:
                model.load_state_dict(torch.load(cfg_model["pretrained_path"], map_location=device))
            else:
                #model = torch.load(cfg_model["pretrained_path"], map_location=device)
                model = torch.nn.DataParallel(model)
        model = model.module
        if cfg_model["freeze"]:
            freeze_layer_parameters(model=model, freeze_layers=cfg_model["freeze_layers"])
        num_ftrs = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv3d(num_ftrs, num_classes, padding=0, kernel_size=1)  #numero di classi 2
    elif model_name == "SFCN2":
        model = sfcn.SFCN2(output_dim=num_classes, fc_size=cfg_model["fc_size"])
        model = torch.nn.DataParallel(model)
        if cfg_model["pretrained"]:
            pretrained_dict = torch.load(cfg_model["pretrained_path"], map_location=device)
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
        model = model.module
        if cfg_model["freeze"]:
            freeze_layer_parameters(model=model, freeze_layers=cfg_model["freeze_layers"])
    elif model_name.startswith("resnet"):
        if model_name == "resnet10":
            model = resnet.resnet10(spatial_dims=3, num_classes=num_classes)
        elif model_name == "resnet18":
            model = resnet.resnet18(spatial_dims=3, num_classes=num_classes)
        elif model_name == "resnet34":
            model = resnet.resnet34(spatial_dims=3, num_classes=num_classes)
        elif model_name == "resnet50":
            model = resnet.resnet50(spatial_dims=3, num_classes=num_classes)
        elif model_name == "resnet101":
            model = resnet.resnet101(spatial_dims=3, num_classes=num_classes)
        elif model_name == "resnet152":
            model = resnet.resnet152(spatial_dims=3, num_classes=num_classes)
        elif model_name == "resnet200":
            model = resnet.resnet200(spatial_dims=3, num_classes=num_classes)
        elif model_name == "resnetfc10":
            model = resnet.resnetfc10(spatial_dims=3, num_classes=num_classes)
        elif model_name == "resnetfc18":
            model = resnet.resnetfc18(spatial_dims=3, num_classes=num_classes)
        elif model_name == "resnetfc34":
            model = resnet.resnetfc34(spatial_dims=3, num_classes=num_classes)
        elif model_name == "resnetfc50":
            model = resnet.resnetfc50(spatial_dims=3, num_classes=num_classes)
        elif model_name == "resnetfc101":
            model = resnet.resnetfc101(spatial_dims=3, num_classes=num_classes)
        elif model_name == "resnetfc152":
            model = resnet.resnetfc152(spatial_dims=3, num_classes=num_classes)
        elif model_name == "resnetfc200":
            model = resnet.resnetfc200(spatial_dims=3, num_classes=num_classes)
        else:
            print("Invalid model name, exiting...")
            exit()
        model = torch.nn.DataParallel(model)
        if cfg_model["pretrained"]:
            if state_dict:
                pretrained_dict = torch.load(cfg_model["pretrained_path"], map_location=device)["state_dict"]
            else:
                pretrained_dict = torch.load(cfg_model["pretrained_path"], map_location=device).state_dict()
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
        model = model.module
        if cfg_model["freeze"]:
            freeze_layer_parameters(model=model, freeze_layers=cfg_model["freeze_layers"])
    elif "densenet" in model_name:
        if model_name == "densenet121":#try this one first pretrained ==False
            model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "densenet169":
            model = monai.networks.nets.DenseNet169(spatial_dims=3, in_channels=1, out_channels=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "densenet201":
            model = monai.networks.nets.DenseNet201(spatial_dims=3, in_channels=1, out_channels=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "densenet264":
            model = monai.networks.nets.DenseNet264(spatial_dims=3, in_channels=1, out_channels=num_classes, pretrained=cfg_model["pretrained"])
    elif "efficientnet" in model_name:
        if model_name == "efficientnet-b0":#try this one first
            model = monai.networks.nets.EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "efficientnet-b1":
            model = monai.networks.nets.EfficientNetBN("efficientnet-b1", spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "efficientnet-b2":
            model = monai.networks.nets.EfficientNetBN("efficientnet-b2", spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "efficientnet-b3":
            model = monai.networks.nets.EfficientNetBN("efficientnet-b3", spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "efficientnet-b4":
            model = monai.networks.nets.EfficientNetBN("efficientnet-b4", spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "efficientnet-b5":
            model = monai.networks.nets.EfficientNetBN("efficientnet-b5", spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "efficientnet-b6":
            model = monai.networks.nets.EfficientNetBN("efficientnet-b6", spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "efficientnet-b7":
            model = monai.networks.nets.EfficientNetBN("efficientnet-b7", spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "efficientnet-b8":
            model = monai.networks.nets.EfficientNetBN("efficientnet-b8", spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "efficientnet-l2":
            model = monai.networks.nets.EfficientNetBN("efficientnet-l2", spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
    elif model_name.startswith("se"):
        if model_name == "senet154":
            model = monai.networks.nets.SENet154(spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "seresnet50": #try this one first
            model = monai.networks.nets.SEResNet50(spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "seresnet101":
            model = monai.networks.nets.SEResNet101(spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "seresnet152":
            model = monai.networks.nets.SEResNet152(spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "seresnext50": #try this one first
            model = monai.networks.nets.SEResNeXt50(spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
        elif model_name == "seresnext101":
            model = monai.networks.nets.SEResNext101(spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=cfg_model["pretrained"])
    #elif model_name == "highresnet": #todo: highresnet
        #model = torch.hub.load('fepegar/highresnet', 'highres3dnet', pretrained=True, in_channels=1, out_channels=1)
    elif model_name == "FusionNetwork":
        network_name = resnet.resnet18(spatial_dims=3, num_classes=num_classes)
        model = fusion.FusionNetwork(network_name)
    else:
        print("Invalid model name, exiting...")
        exit()


    return model


def initialize_joint_model(model_name, num_classes, cfg_model, fold, device):
    if model_name in ["resnet50SFCN", "resnet50SFCN_prob", "resnet50SFCN_prob_multi"]:
        # Single Models
        if cfg_model['pretrained_1']:
            cfg_model['pretrained'] = True
            if os.path.isdir(cfg_model['pretrained_path_1']):
                cfg_model['pretrained_path'] = os.path.join(cfg_model['pretrained_path_1'], str(fold), "%s.pt" % cfg_model['model_name_1'])
                state_dict = False
            if os.path.isfile(cfg_model['pretrained_path_1']):
                cfg_model['pretrained_path'] = cfg_model['pretrained_path_1']
                state_dict = True
        else:
            cfg_model['pretrained'] = False
            cfg_model['pretrained_path'] = None
            state_dict = False
        model_1 = initialize_model(model_name=cfg_model['model_name_1'], num_classes=num_classes, cfg_model=cfg_model, device=device, state_dict=state_dict)
        if cfg_model['pretrained_2']:
            cfg_model['pretrained'] = True
            if os.path.isdir(cfg_model['pretrained_path_2']):
                cfg_model['pretrained_path'] = os.path.join(cfg_model['pretrained_path_2'], str(fold), "%s.pt" % cfg_model['model_name_2'])
                state_dict = False
            if os.path.isfile(cfg_model['pretrained_path_2']):
                cfg_model['pretrained_path'] = cfg_model['pretrained_path_2']
                state_dict = True
        else:
            cfg_model['pretrained'] = False
            cfg_model['pretrained_path'] = None
            state_dict = False
        model_2 = initialize_model(model_name=cfg_model['model_name_2'], num_classes=num_classes, cfg_model=cfg_model, device=device, state_dict=state_dict)
        # Joint Models
        if model_name == "resnet50SFCN":
            model = resnetsfcn.ResNetSFCN(model_1, model_2, num_classes)
        if model_name == "resnet50SFCN_prob":
            model = resnetsfcn.ResNetSFCN_Prob(model_1, model_2, num_classes)
        if model_name == "resnet50SFCN_prob_multi":
            model = resnetsfcn.ResNetSFCN_Prob_Multi(model_1, model_2, num_classes)
    else:
        print("Invalid model name, exiting...")
        exit()

    return model


def get_train_function(output_type):
    if output_type == 'single':
        return train_model
    if output_type == 'multi':
        return train_model_multi
    else:
        print("Invalid train type, exiting...")
        exit()


def train_model(model, criterion, optimizer, scheduler, model_name, data_loaders, model_dir, device, cfg_trainer):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    epochs_no_improve = 0
    early_stop = False

    for epoch in range(cfg_trainer["max_epochs"]):
        #break
        print('Epoch {}/{}'.format(epoch, cfg_trainer["max_epochs"] - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{cfg_trainer["max_epochs"]}', unit='img') as pbar:
                for inputs, labels, file_names in data_loaders[phase]:
                    #break
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs.float())    #due input

                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    pbar.update(inputs.shape[0])

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)


            if phase == 'val':
                scheduler.step(epoch_loss)

            # update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Early stopping
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= cfg_trainer["early_stopping"]:
                        print(f'\nEarly Stopping! Total epochs: {epoch}%')
                        early_stop = True
                        break

        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save model
    torch.save(model, os.path.join(model_dir, "%s.pt" % model_name))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return model, history


def train_model_fusion(model, criterion, optimizer, scheduler, model_name, data_loaders, model_dir, device, cfg_trainer):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    epochs_no_improve = 0
    early_stop = False

    for epoch in range(cfg_trainer["max_epochs"]):
        print('Epoch {}/{}'.format(epoch, cfg_trainer["max_epochs"] - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{cfg_trainer["max_epochs"]}', unit='img') as pbar:
                for inputs, labels, file_names in data_loaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs_1, outputs_2, outputs_3 = model(inputs.float())
                        _, preds = torch.max(outputs_3, 1)
                        loss_1 = criterion(outputs_1, labels)
                        loss_2 = criterion(outputs_2, labels)
                        loss_3 = criterion(outputs_3, labels)
                        loss = loss_1 + loss_2 + loss_3
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    pbar.update(inputs.shape[0])

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

            if phase == 'val':
                scheduler.step(epoch_loss)

            # update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Early stopping
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= cfg_trainer["early_stopping"]:
                        print(f'\nEarly Stopping! Total epochs: {epoch}%')
                        early_stop = True
                        break

        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save model
    torch.save(model, os.path.join(model_dir, "%s.pt" % model_name))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return model, history


def plot_training(history, model_name, plot_training_dir):
    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)
    # Training results Loss function
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'val_loss']:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(model_plot_dir, "Loss"))
    # Training results Accuracy
    plt.figure(figsize=(8, 6))
    for c in ['train_acc', 'val_acc']:
        plt.plot(100 * history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(model_plot_dir, "Acc"))


def get_evaluate_function(output_type):
    if output_type == 'single':
        return evaluate
    if output_type == 'multi':
        return evaluate_multi
    else:
        print("Invalid train type, exiting...")
        exit()


def evaluate(model, data_loader, device):
    # Global and Class Accuracy
    correct_pred = {classname: 0 for classname in list(data_loader.dataset.idx_to_class.values()) + ["all"]}
    total_pred = {classname: 0 for classname in list(data_loader.dataset.idx_to_class.values()) + ["all"]}

    # Test loop
    model.eval()
    with torch.no_grad():
        for inputs, labels, file_names in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Prediction
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)
            # global
            correct_pred['all'] += (preds == labels).sum().item()
            total_pred['all'] += labels.size(0)
            # class
            for label, prediction in zip(labels, preds):
                if label == prediction:
                    correct_pred[data_loader.dataset.idx_to_class[label.item()]] += 1
                total_pred[data_loader.dataset.idx_to_class[label.item()]] += 1

    # Accuracy
    test_results = {k: correct_pred[k]/total_pred[k] for k in correct_pred.keys() & total_pred}

    return test_results


def evaluate_multi(model, data_loader, device):
    # Global and Class Accuracy
    correct_pred = {classname: 0 for classname in list(data_loader.dataset.idx_to_class.values()) + ["all"]}
    total_pred = {classname: 0 for classname in list(data_loader.dataset.idx_to_class.values()) + ["all"]}

    # Test loop
    model.eval()
    with torch.no_grad():
        for inputs, labels, file_names in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Prediction
            outputs_1, outputs_2, outputs_3 = model(inputs.float())
            _, preds = torch.max(outputs_3, 1)
            # global
            correct_pred['all'] += (preds == labels).sum().item()
            total_pred['all'] += labels.size(0)
            # class
            for label, prediction in zip(labels, preds):
                if label == prediction:
                    correct_pred[data_loader.dataset.idx_to_class[label.item()]] += 1
                total_pred[data_loader.dataset.idx_to_class[label.item()]] += 1

    # Accuracy
    test_results = {k: correct_pred[k]/total_pred[k] for k in correct_pred.keys() & total_pred}

    return test_results


def get_predict_function(output_type='single'):
    if output_type == 'single':
        return predict
    if output_type == 'multi':
        return predict_multi
    else:
        print("Invalid train type, exiting...")
        exit()


def predict(model, data_loader, device):
    # Prediction and Truth
    predictions = {}
    probabilities = {}
    truth = {}

    # Predict loop
    model.eval()
    with torch.no_grad():
        for inputs, labels, file_names in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Prediction
            outputs = model(inputs.float())
            probs = nn.functional.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)
            for file_name, label, pred, prob in zip(file_names, labels, preds, probs):
                predictions[file_name] = pred.item()
                probabilities[file_name] = prob.tolist()
                truth[file_name] = label.item()
    return predictions, probabilities, truth


def predict_multi(model, data_loader, device):
    # Prediction and Truth
    predictions = {}
    probabilities = {}
    truth = {}

    # Predict loop
    model.eval()
    with torch.no_grad():
        for inputs, labels, file_names in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Prediction
            outputs_1, outputs_2, outputs_3 = model(inputs.float())
            probs = nn.functional.softmax(outputs_3, 1)
            _, preds = torch.max(outputs_3, 1)
            for file_name, label, pred, prob in zip(file_names, labels, preds, probs):
                predictions[file_name] = pred.item()
                probabilities[file_name] = prob.tolist()
                truth[file_name] = label.item()
    return predictions, probabilities, truth


def get_predictions(prediction_dir, fold_list, steps):
    results = pd.DataFrame()

    for fold in fold_list:

        fold_path = os.path.join(prediction_dir, str(fold))

        for Fset in steps:

            preds_path = os.path.join(fold_path, f"prediction_{Fset}_{fold}.xlsx")
            probs_path = os.path.join(fold_path, f"probability_{Fset}_{fold}.xlsx")

            preds = pd.read_excel(preds_path, engine="openpyxl", index_col=0)
            probs = pd.read_excel(probs_path, engine="openpyxl", index_col=0)

            preds.index.name = 'ID'
            probs.index.name = 'ID'

            cols_to_drop = [col for col in probs.columns.to_list() if col.endswith("_0")]
            probs = probs.drop(cols_to_drop + ["True"], axis=1)

            clear_names = [col.replace("_1", "") for col in probs.columns.to_list()]
            new_names = pd.MultiIndex.from_product([clear_names, ["probability"]])

            probs.columns = new_names

            new_names = pd.MultiIndex.from_product([preds.columns.to_list(), ["prediction"]])

            preds.columns = new_names

            preds = pd.concat([preds, probs], axis=1)

            for classifier in preds.columns.levels[0]:
                preds[(classifier, "label")] = preds[("True", "prediction")]

            preds = preds.drop([("True", "label"), ("True", "prediction")], axis=1)

            preds = preds.assign(fold=fold, Fset=Fset)

            preds = preds.reset_index().set_index(["Fset", "fold", "ID"])

            results = pd.concat([results, preds], axis=0)

    return results.sort_index(axis=1)


def compute_performance(results, performance_dir, step):
    results = results.set_index("ID")

    #TP = sum(pd.DataFrame([results.label == 1, results.prediction == 1]).all(axis=0))
    #TN = sum(pd.DataFrame([results.label == 0, results.prediction == 0]).all(axis=0))
    #FP = sum(pd.DataFrame([results.label == 0, results.prediction == 1]).all(axis=0))
    #FN = sum(pd.DataFrame([results.label == 1, results.prediction == 0]).all(axis=0))
    TP = sum(pd.DataFrame([results.label == 0, results.prediction == 0]).all(axis=0))
    TN = sum(pd.DataFrame([results.label == 1, results.prediction == 1]).all(axis=0))
    FP = sum(pd.DataFrame([results.label == 1, results.prediction == 0]).all(axis=0))
    FN = sum(pd.DataFrame([results.label == 0, results.prediction == 1]).all(axis=0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TN + FP == 0:
        specificity = 0
    else:
        specificity = TN / (TN + FP)

    auc = roc_auc_score(results.label, results.probability)

    fscore = TP / (TP + 0.5 * (FP + FN))

    # all0 = sum(results.label == 0)/results.shape[0]
    # all1 = sum(results.label == 1)/results.shape[0]

    performance = {"AUC": auc, "accuracy": accuracy, "recall": recall, "precision": precision, "specificity": specificity, "fscore": fscore}

    performance.update((metric, [round(value * 100, 2)]) for metric, value in performance.items())

    # ROC Curve
    fpr, tpr, t = roc_curve(results.label, results.probability)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.savefig(os.path.join(performance_dir, 'roc_curve_%s' % step), dpi=300)

    # Error-Reject Curve
    reject = []
    error = []
    for t in np.arange(0.5, 1, 0.01):
        t_inv = 1 - t
        results_reject = results[(results.probability >= t) | (results.probability <= t_inv)]
        reject.append((len(results.probability) - len(results_reject)) / len(results.probability))
        wrong = results_reject.probability.round() != results_reject.label
        error.append(wrong.sum() / len(results_reject))
    sns.lineplot(x=reject, y=error, markers=True)
    plt.xlabel("Reject Rate")
    plt.ylabel("Error Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(performance_dir, 'er_curve_%s' % step), dpi=300)

    return pd.DataFrame(performance)


def get_performance(results, performance_dir):
    results = results.reorder_levels([1, 0], axis=1).stack().reset_index()
    results = results.rename({"level_3": "classifier"}, axis=1)

    nFolds = results.fold.nunique()
    for fold in range(nFolds):
        performance_fold_dir = os.path.join(performance_dir, str(fold))
        util_general.create_dir(performance_fold_dir)

    performance_per_fold = results.groupby(by=["Fset", "fold", "classifier"]).apply(lambda x: compute_performance(x, performance_dir=os.path.join(performance_dir, str(x.name[1])), step=x.name[0])).droplevel(3)

    average_performance = performance_per_fold.reset_index().drop("fold", axis=1).groupby(by=["Fset", "classifier"]).agg(["mean", "std"]).round(2)

    performance_per_fold = performance_per_fold.reorder_levels([0, 2, 1], axis=0).unstack()

    performance = pd.concat([performance_per_fold, average_performance], axis=1).reorder_levels([1, 0], axis=1).sort_index(axis=1)

    performance = performance[["mean", "std"] + list(range(nFolds))]

    # Save
    performance.to_excel(os.path.join(performance_dir, "performance.xlsx"))

    return performance










#train model Fusion

def train_model_fusion(model, criterion_1, criterion_2, optimizer, scheduler, model_name, data_loaders, model_dir, device, cfg_trainer, cfg_batch_size):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    epochs_no_improve = 0
    early_stop = False

    for epoch in range(cfg_trainer["max_epochs"]):
        #break
        print('Epoch {}/{}'.format(epoch, cfg_trainer["max_epochs"] - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_corrects = 0

            running_loss_1 = 0.0
            running_loss_2 = 0.0

            # Iterate over data
            with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{cfg_trainer["max_epochs"]}', unit='img') as pbar:

                for inputs1, inputs2, labels, file_names in data_loaders[phase]:
                    #break
                    inputs1 = inputs1.to(device)  #T1
                    labels = labels.to(device)  #T2

                    inputs2 = inputs2.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs1, outputs2 = model(inputs1.float(), inputs2.float())    #due input

                        loss1 = criterion_1(outputs1, labels)
                        loss2 = criterion_2(outputs2, labels)

                        loss = loss1 + loss2
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        softmax = nn.Softmax(dim=1)

                        outputs1 = softmax(outputs1)
                        outputs2 = softmax(outputs2)

                        for i in range(0, int(cfg_batch_size)):
                            out = [outputs1[i], outputs2[i]]
                            media0 = ((out[i][0] + out[i + 1][0]) /2)
                            media1 = ((out[i][1] + out[i + 1][1]) /2)
                            if media0 > media1:
                                preds = 0
                            else:
                                preds = 1

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss_1 += loss.item() * inputs1.size(0)
                    running_loss_2 += loss.item() * inputs2.size(0)

                    running_loss = running_loss_1 + running_loss_2

                    running_corrects += torch.sum(preds == labels.data)

                    pbar.update(inputs1.shape[0])
                    pbar.update(inputs2.shape[0])

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)


            if phase == 'val':
                scheduler.step(epoch_loss)

            # update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Early stopping
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= cfg_trainer["early_stopping"]:
                        print(f'\nEarly Stopping! Total epochs: {epoch}%')
                        early_stop = True
                        break

        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save model
    torch.save(model, os.path.join(model_dir, "%s.pt" % model_name))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return model, history


def evaluate_fusion(model, data_loader, device, cfg_batch_size):

    # Global and Class Accuracy
    correct_pred = {classname: 0 for classname in list(data_loader.dataset.idx_to_class.values()) + ["all"]}
    total_pred = {classname: 0 for classname in list(data_loader.dataset.idx_to_class.values()) + ["all"]}

    # Test loop
    model.eval()
    with torch.no_grad():
        for inputs1, inputs2, labels, file_names in tqdm(data_loader):
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)
            # Prediction
            outputs1 = model(inputs1.float())
            outputs2 = model(inputs2.float())

            softmax = nn.Softmax(dim=1)

            outputs1 = softmax(outputs1)
            outputs2 = softmax(outputs2)

            for i in range(0, int(cfg_batch_size)):
                out = [outputs1[i], outputs2[i]]
                media0 = (out[i][0] + out[i + 1][0]) / 2
                media1 = (out[i][1] + out[i + 1][1]) / 2
                if media0 > media1:
                    preds = 0
                else:
                    preds = 1

            # global
            correct_pred['all'] += (preds == labels).sum().item()
            total_pred['all'] += labels.size(0)
            # class
            for label, prediction in zip(labels, preds):
                if label == prediction:
                    correct_pred[data_loader.dataset.idx_to_class[label.item()]] += 1
                total_pred[data_loader.dataset.idx_to_class[label.item()]] += 1

    # Accuracy
    test_results = {k: correct_pred[k] / total_pred[k] for k in correct_pred.keys() & total_pred}

    return test_results