'''
/* ===========================================================================
** Copyright (C) 2019 Infineon Technologies AG. All rights reserved.
** ===========================================================================
**
** ===========================================================================
** Infineon Technologies AG (INFINEON) is supplying this file for use
** exclusively with Infineon's sensor products. This file can be freely
** distributed within development tools and software supporting such 
** products.
** 
** THIS SOFTWARE IS PROVIDED "AS IS".  NO WARRANTIES, WHETHER EXPRESS, IMPLIED
** OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
** MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE.
** INFINEON SHALL NOT, IN ANY CIRCUMSTANCES, BE LIABLE FOR DIRECT, INDIRECT, 
** INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES, FOR ANY REASON 
** WHATSOEVER.
** ===========================================================================
*/
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
# ========================= File Paths =============================
    parser.add_argument('--root_path', default='', type=str, help='Root directory path of data')
    parser.add_argument('--video_path', default='/home/root5/GeScale/SHGD/SHGD_Single', type=str, help='Directory path of Videos')
    parser.add_argument('--annotation_path', default='/home/root5/GeScale/SHGD/SHGD_Single/annotation-SHGD13', type=str, help='Annotation file path')
    parser.add_argument('--train_list', default='SHGD_13_trainlist.txt', type=str, help='File name of train list')
    parser.add_argument('--val_list', default='SHGD_13_vallist.txt', type=str, help='File name of validation list')
    parser.add_argument('--test_list', default='test_list.txt', type=str, help='File name of test list')
    parser.add_argument('--result_path', default='/home/root5/GeScale/choyaa-GeScale-master/my_mobilenetv2_pruning_two/results/SHGD13_prune', type=str, help='Result directory path')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--pretrain_path', default='/home/root5/GeScale/choyaa-GeScale-master/my_mobilenetv2_pruning_two/results/SHGD13_sparsity/SHGD_mobilenetv2_IRD_8_checkpoint.pth', type=str, help='Pretrained model (.pth)')
#resumepath         /home/root5/GeScale/choyaa-GeScale-master/my__netslimming+3D/results/SHGD13_sparsity/SHGD_mobilenetv2_IRD_8_sparsity_checkpoint.pth  
# ========================= Model Configs ==========================
    parser.add_argument('--model', default='mobilenetv2', type=str, help='(squeezenet1_1 | mobilenetv2 ')
    parser.add_argument('--version', default=1.1, type=float, help='Version of the model')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of the model')

    parser.add_argument('--store_name', default='SHGD13_mobilenetv2_IRD_8_prune_checkpoint', type=str, help='Name to store checkpoints')
    parser.add_argument('--modality', default='IRD', type=str, help='Modality of input data. RGB, IR, D or IRD')
    parser.add_argument('--dataset', default='SHGD', type=str, help='Used dataset ( jester | SHGD)')
    parser.add_argument('--n_classes', default=13, type=int, help='Number of classes (jester: 27, SHGD: 15)')
    parser.add_argument('--n_finetune_classes', default=13, type=int, help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    parser.add_argument('--same_modality_finetune', action='store_true', help='If true, finetuning modality is the same as pretraining.')
    parser.set_defaults(same_modality_finetune=True)
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=8, type=int, help='Temporal duration of inputs')

# ========================= Training Configs ==========================
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=60, type=int, help='Number of total epochs to run')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--lr_steps', default=[30,45,55], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--train_crop', default='random', type=str, help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--mean_dataset', default='activitynet', type=str, help='dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument('--no_mean_norm', action='store_true', help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm', action='store_true', help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--optimizer', default='sgd', type=str, help='Currently only support SGD')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--n_val_samples', default=3, type=int, help='Number of validation samples for each activity')
    parser.add_argument('--ft_begin_index', default=0, type=int, help='Begin block index of fine-tuning')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint', default=10, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--norm_value', default=1, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--width_mult', default=1.0, type=float, help='The applied width multiplier to scale number of filters')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')

# ========================= Training/Val/Test Mode ==========================
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)

#=========================sparsity ============================================
    parser.add_argument('--sr',  action='store_true',help='train with channel sparsity regularization')
    parser.set_defaults(sparsity_regularization=False)
    parser.add_argument('--s', type=float, default=0.0001,
                        help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--refine', default='', type=str, metavar='PATH',
                        help='refine from prune model')


#========================= prune ===============================================
    parser.add_argument('--percent', type=float, default=0.8,
                        help='scale sparse rate (default: 0.5)')



    args = parser.parse_args()

    return args
