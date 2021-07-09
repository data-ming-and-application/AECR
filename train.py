"""
# Pytorch implementation for paper
# "AECR: Alignment Efficient Cross-Modal Retrieval Considering Transferable Representation Learning"
# Yang Yang, Jinyi Guo, Hengshu Zhu, Dianhai Yu, Fuzhen Zhuang, Hui Xiong and Jian Yang
"""

import os
import time
import shutil

import torch
import numpy

import data
import opts
from vocab import Vocabulary, deserialize_vocab
from model import AECR
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_attn_scores,evalrank

import logging
import tensorboard_logger as tb_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    opt = opts.parse_opt()
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    vocab = deserialize_vocab(opt.vocab_path)
    opt.vocab_size = len(vocab)

    # Load data loaders
    train_loader_source, val_loader_source = data.get_loaders('source',
                                                              opt.data_name_source, vocab, opt.batch_size_source,
                                                              opt.workers, opt)
    train_loader_target = data.get_loaders('target',
                                           opt.data_name_target, vocab, opt.batch_size_target, opt.workers, opt)
    # Construct the model
    model = AECR(opt)

    # Train the Model
    best_rsum = 0
    start_epoch = 0
    # optionally resume from a checkpoint
    if opt.model_path:
        if os.path.isfile(opt.model_path):
            print("=> loading checkpoint '{}'".format(opt.model_path))
            checkpoint = torch.load(opt.model_path)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'], opt.mode)
            # Eiters is used to show logs as the continuation of another
            model.Eiters = 0
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.model_path, start_epoch, best_rsum))
        else:
            print("=> no checkpoint found at '{}'".format(opt.model_path))

    #Train the model
    for epoch in range(0,opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)

        adjust_learning_rate(opt, model.optimizer_enc, epoch)
        adjust_learning_rate(opt, model.optimizer_critic, epoch)

        # train for one epoch
        train(opt, train_loader_source,train_loader_target, model, epoch)

        # remember best R@ sum and save checkpoint

        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            #'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, False, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')


def train(opt, train_loader_source, train_loader_target, model, epoch):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    train_loader_source_iter = iter(train_loader_source)
    for i, train_data_target in enumerate(train_loader_target):
        try:
            train_data_source = next(train_loader_source_iter)
        except StopIteration:
            train_loader_source_iter = iter(train_loader_source)
            train_data_source=next(train_loader_source_iter)

        # switch to train mode
        model.train_start()
        images_n, captions_n, lengths_n, ids_n=train_data_target
        images_p, captions_p, lengths_p, ids_p=train_data_source

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger
        
        # Update the model
        model.train_emb(images_n, captions_n, lengths_n, ids_n,
                  images_p, captions_p, lengths_p, ids_p)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader_target), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def validate(opt, val_loader, model, epoch):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)

    # clear duplicate 5*images and keep 1*images
    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    # record computation time of validation
    start = time.time()
    sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100)
    end = time.time()
    print("calculate similarity time:", end-start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanr))

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r_sum', r_sum, step=model.Eiters)

    return r_sum


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR
    decayed by 10 after opt.lr_update epoch
    """
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
    evalrank(model_path='', data_path='./data', split='test', fold5=False)