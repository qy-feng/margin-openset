from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer

from data.dataloader import *
from data.centroid import *
from model import Net
import utils


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu))
        is_cuda = True
    else:
        device = torch.device("cpu")
        is_cuda = False

    src_loader, tgt_loader = get_data(args)

    model = Net(task=args.task).to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
                            
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))

    best_acc = 0
    best_label = []
    best_result = []

    # create centroids for known classes
    all_centroids = Centroids(args.class_num - 1, 100, use_cuda=is_cuda)

    try:
        # start training
        for epoch in range(args.epochs):
            data = (src_loader, tgt_loader, all_centroids)

            all_centroids = train(model, optimizer, data, epoch, device, args)

            result, gt_label, acc = test(model, tgt_loader, epoch, device, args)

            is_best = acc > best_acc
            if is_best:
                best_acc = acc
                best_label = gt_label
                best_pred = result

            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc
            }, is_best, args.check_dir)

        print ("------Best-------")
        utils.cal_acc(best_label, best_result, args.class_num)

    except KeyboardInterrupt:
        print ("------Best-------")
        utils.cal_acc(best_label, best_result, args.class_num)


def train(model, optimizer, data, epoch, device, args):

    src_loader, tgt_loader, all_centroids = data
    pre_stage = 5
    adv_stage = 15
    criterion_bce = nn.BCELoss()
    criterion_cel = nn.CrossEntropyLoss()

    model.train()

    for batch_idx, (batch_s, batch_t) in enumerate(zip(src_loader, tgt_loader)):
        global_step = epoch * len(src_loader) + batch_idx
        p = global_step / args.epochs * len(src_loader)
        lr = utils.adjust_learning_rate(optimizer, epoch, args,
                                   batch_idx, len(src_loader))
        data_s, label_s = batch_s
        data_s = data_s.to(device)
        label_s = label_s.to(device)
        data_t, label_t = batch_t
        data_t = data_t.to(device)
        adv_label_t = torch.tensor([args.th]*len(data_t)).to(device)

        loss = 0
        optimizer.zero_grad()
        feat_s, pred_s = model(data_s)
        feat_t, pred_t = model(data_t, p, adv=True)
        
        # classification loss for known classes in source domain
        loss_cel = criterion_cel(pred_s, label_s)
        loss += loss_cel

        if epoch >= pre_stage:
            # adversarial loss for unknown class in target domain
            pred_t_prob_unk = F.softmax(pred_t, dim=1)[:, -1]
            loss_adv = criterion_bce(pred_t_prob_unk, adv_label_t)
            loss += loss_adv
        
        if epoch >= adv_stage:
            all_centroids.update(feat_s, pred_s, label_s, feat_t, pred_t)
            s_ctds, t_ctds = all_centroids.get_centroids() 

            loss_intra = crit_intra(feat_s, label_s, s_ctds)
            loss += loss_intra * args.lamb_s

            loss_inter, _ = crit_inter(s_ctds, t_ctds)
            loss += loss_inter * args.lamb_c

            loss_contr = crit_contrast(feat_t, pred_t, s_ctds, t_ctds)
            loss += loss_contr * args.lamb_t

        loss.backward()
        optimizer.step()

        if epoch >= pre_stage and batch_idx % args.log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]  LR: {:.6f} \
                   Loss(cel): {:.4f}  Loss(adv): {:.4f}\t'.format(
                        epoch, batch_idx * args.batch_size, 
                        len(src_loader.dataset),
                        100. * batch_idx / len(src_loader), lr, 
                        loss_cel.item(), loss_adv.item()))
        
    return all_centroids


def test(model, tgt_loader, epoch, device, args):
    
    loss = 0
    correct = 0
    result = []
    gt_label = []

    model.eval()
    criterion_cel = nn.CrossEntropyLoss()
    
    for batch_idx, (data_t, label) in enumerate(tgt_loader):
        data_t = data_t.to(device)
        label = label.to(device)

        feat, output = model(data_t)
        pred = output.max(1, keepdim=True)[1]
        loss += criterion_cel(output, label).item()

        for i in range(len(pred)):
            result.append(pred[i].item())
            gt_label.append(label[i].item())

        correct += pred.eq(label.view_as(pred)).sum().item()

    loss /= len(tgt_loader.dataset)

    utils.cal_acc(gt_label, result, args.class_num)
    acc = 100. * correct / len(tgt_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(tgt_loader.dataset),
        100. * correct / len(tgt_loader.dataset)))

    return result, gt_label, acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Openset-DA SVHN -> MNIST Example')
    parser.add_argument('--task', choices=['s2m', 'u2m', 'm2u'], default='s2m',
                        help='domain adaptation sub-task')
    parser.add_argument('--class-num', type=int, default=6, help='number of classes')
    parser.add_argument('--th', type=float, default=0.5, metavar='TH',
                        help='threshold for unknown class')
    parser.add_argument('--lamb-s', type=float, default=0.02)
    parser.add_argument('--lamb-c', type=float, default=0.005)
    parser.add_argument('--lamb-t', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr-rampdown-epochs', default=101, type=int,
                            help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    # parser.add_argument('--grl-rampup-epochs', default=20, type=int, metavar='EPOCHS',
    #                         help='length of grl rampup')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                        help='weight decay (default: 1e-3)')

    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--check_dir', default='checkpoint', type=str,
                        help='directory to save checkpoint')
    parser.add_argument('--resume', default='', type=str,
                        help='path to resume checkpoint (default: none)')
    parser.add_argument('--gpu', default='0', type=str, metavar='GPU',
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)