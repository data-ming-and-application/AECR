"""Argument parser"""

import argparse


def parse_opt():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # --------------------------- data path -------------------------#
    parser.add_argument('--data_path', default='./data',
                        help='path to datasets')
    parser.add_argument('--data_name_source', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--data_name_target', default='coco_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./vocab/common_precomp_vocab.json',
                        help='Path to saved common vocabulary json.')
    parser.add_argument('--model_name', default='./runs/coco',
                        help='Path to save the model.')
    parser.add_argument('--logger_name', default='./runs/coco/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--mode', default='train', type=str,
                        help='{train, resume}')
    parser.add_argument('--model_path', default='./pretrain/SGRAF_f30k_model_best.pth.tar', type=str,
                        help='path to load pretrained model')

    # ----------------------- training setting ----------------------#
    parser.add_argument('--batch_size_source', default=56, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--batch_size_target', default=56, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--num_epochs', default=3, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', default=20, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--learning_rate', default=.0001, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_false',
                        help='Use max instead of sum in the rank loss.')

    # ------------------------- model setting -----------------------#
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--sim_dim', default=256, type=int,
                        help='Dimensionality of the sim embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--bi_gru',default=True,action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--module_name', default='SAF', type=str,
                        help='SGR, SAF')
    parser.add_argument('--sgr_step', default=3, type=int,
                        help='Step of the SGR.')
    parser.add_argument('--u', default=1, type=float,
                        help='triplet loss weight.')
    parser.add_argument('--lamda', default=0.001, type=float,
                        help='structure transfer loss weight')

    opt = parser.parse_args()
    print(opt)
    return opt
