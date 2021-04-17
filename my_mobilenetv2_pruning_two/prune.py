import os
import torch
from torchvision import datasets, transforms
from model import generate_model
from opts import parse_opts
import json
from os.path import join
from dataset import get_training_set, get_validation_set, get_test_set
import data_loader
from mean import get_mean, get_std
from torch.optim import lr_scheduler
from torch import optim
from Slimming import SlimmingPrune


opt_prune = parse_opts()
#opt_prune.pretrain_path = '/home/root5/GeScale/choyaa-GeScale-master/my__netslimming+3D/results/SHGD13_sparsity/SHGD_mobilenetv2_IRD_8_best.pth'

#===========initialize
opt_prune.scales = [opt_prune.initial_scale]
for i in range(1, opt_prune.n_scales):
    opt_prune.scales.append(opt_prune.scales[-1] * opt_prune.scale_step)
# opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
opt_prune.arch = '{}'.format(opt_prune.model)
opt_prune.mean = get_mean(opt_prune.norm_value, dataset=opt_prune.mean_dataset)
opt_prune.std = get_std(opt_prune.norm_value)

opt_prune.store_name = '_'.join([opt_prune.dataset, opt_prune.model,
                                 opt_prune.modality, str(opt_prune.sample_duration)])

torch.manual_seed(opt_prune.manual_seed)  # 生成固定随机数

criterion, norm_method= data_loader.get_criterion_and_norm(opt_prune)

#==========generate model
model, parameters = generate_model(opt_prune)   #if opt_prune.pretrain_path , 预装模型初始化和加载
newmodel, newparameters = generate_model(opt_prune)
'''
#===========train dataset
if not opt_prune.no_train:
    train_loader, train_logger, train_batch_logger = data_loader.get_traininfo(opt_prune, norm_method)

if opt_prune.nesterov:
    dampening = 0
else:
    dampening = opt_prune.dampening
#====optimizer , scheduler
optimizer = optim.SGD(
        parameters,
        lr=opt_prune.learning_rate,
        momentum=opt_prune.momentum,
        dampening=dampening,
        weight_decay=opt_prune.weight_decay,
        nesterov=opt_prune.nesterov)
scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=opt_prune.lr_patience)
#==========validation dataset
if not opt_prune.no_val:
    validation_data, val_loader, val_logger = data_loader.get_valinfo(opt_prune, norm_method)
#===========test dataset
test_loader = data_loader.get_testinfo(opt_prune, norm_method)
'''
test_loader= []
train_loader = []
if opt_prune.nesterov:
    dampening = 0
else:
    dampening = opt_prune.dampening
#====optimizer , scheduler
optimizer = optim.SGD(
        parameters,
        lr=opt_prune.learning_rate,
        momentum=opt_prune.momentum,
        dampening=dampening,
        weight_decay=opt_prune.weight_decay,
        nesterov=opt_prune.nesterov)
#++++++++++++++++开始！+++++++
#赋值给slimming 中 BasePruner(通过子类SlimmingPruner)
pruner = SlimmingPrune(model= model, newmodel= newmodel, testset=test_loader, trainset=train_loader, optimizer = optimizer, args=opt_prune)
pruner.prune()




