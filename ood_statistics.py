# compute ece and ood (CIFAR10) for a trained model

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar10_dataloaders
from torchmetrics import CalibrationError
from sklearn.covariance import EmpiricalCovariance
from sklearn import metrics
from tqdm import tqdm
from models import model_dict
import numpy as np
import torch
from torch import nn
from helper.loops import train_distill as train, validate
from netcal.metrics import ECE
from torchmetrics import CalibrationError

ece = ECE(bins=15)
ece_v2 = CalibrationError(n_bins=15)

def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh

def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh

def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)
    return auroc, aupr_in, aupr_out

train_loader, val_loader = get_cifar100_dataloaders(batch_size=32, num_workers=4)

_, ood_loader = get_cifar10_dataloaders(batch_size=32, num_workers=4)

features_tr = []
features_val = []
features_ood = []
labels_tr = []

model = model_dict["wrn_40_2"](num_classes=100).cuda().eval()
model.load_state_dict(torch.load("save/models/wrn_40_2_cifar100_lr_0.05_decay_0.0005_trial_0_deit3aug/wrn_40_2_last.pth")["model"])

acc, _, _ = validate(val_loader, model, nn.CrossEntropyLoss(), None, debug=True)
print(f"model accuracy: {acc}")
ce = CalibrationError()

for idx, (input, target) in enumerate(train_loader):
    input = input.float()
    input = input.cuda()
    target = target.cuda()
    feat, output = model(input, is_feat=True)
    features_tr.append(feat[-1].detach().cpu().numpy())
    labels_tr.append(target.cpu().numpy())

features_tr = np.concatenate(features_tr, axis=0)
labels_tr = np.concatenate(labels_tr, axis=0)

targets_val = []
outputs_val = []
for idx, (input, target) in enumerate(val_loader):
    input = input.float()
    input = input.cuda()
    feat, output = model(input, is_feat=True)
    targets_val.append(target)
    features_val.append(feat[-1].detach().cpu().numpy())
    outputs_val.append(output.detach().cpu().numpy())

targets_val = np.concatenate(targets_val, axis=0)
outputs_val = np.concatenate(outputs_val, axis=0)
features_val = np.concatenate(features_val, axis=0)

def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x))
    sm = e_x / e_x.sum(axis=axis, keepdims=True)
    return sm

def expected_calibration_error(y_true, y_pred, num_bins=15):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]

print(f"Calibration Error (ECE) on CIFAR100-Val: ", ece.measure(softmax(outputs_val), targets_val))
# ece_v2.update(preds=torch.from_numpy(softmax(outputs_val)), target=torch.from_numpy(targets_val))
# print(f"Calibration Error (ECE) on CIFAR100-Val: ", ece_v2.compute())
# print(f"Calibration Error (ECE) on CIFAR100-Val: ", expected_calibration_error(targets_val, softmax(outputs_val)))


outputs_ood = []
for idx, (input, target) in enumerate(ood_loader):
    input = input.float()
    input = input.cuda()
    target = target.cuda()
    feat, output = model(input, is_feat=True)
    features_ood.append(feat[-1].detach().cpu().numpy())
    outputs_ood.append(output.detach().cpu().numpy())

features_ood = np.concatenate(features_ood, axis=0)
outputs_ood = np.concatenate(outputs_ood, axis=0)

print("Computing MSP Score:")
msp_id = np.max(softmax(outputs_val), axis=1)
msp_ood = np.max(softmax(outputs_ood), axis=1)
auc_ood = auc(msp_id, msp_ood)[0]
fpr_ood, _ = fpr_recall(msp_id, msp_ood, 0.95)
print(f"AUC: {auc_ood}, FPR95: {fpr_ood}")

print("Computing Mahalanobis Score:")

train_means = []
train_feat_centered = []
for i in range(100):
    fs = features_tr[labels_tr == i]
    _m = fs.mean(axis=0)
    train_means.append(_m)
    train_feat_centered.extend(fs - _m)

print('computing precision matrix...')
ec = EmpiricalCovariance(assume_centered=True)
ec.fit(np.array(train_feat_centered).astype(np.float64))

print('go to gpu...')
mean = torch.from_numpy(np.array(train_means)).cuda().float()
prec = torch.from_numpy(ec.precision_).cuda().float()

score_id = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in tqdm(torch.from_numpy(features_val).cuda().float())])
score_ood = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in tqdm(torch.from_numpy(features_ood).cuda().float())])
auc_ood = auc(score_id, score_ood)[0]
fpr_ood, _ = fpr_recall(score_id, score_ood, 0.95)
print(f"AUC: {auc_ood}, FPR95: {fpr_ood}")
