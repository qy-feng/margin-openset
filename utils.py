import os
import shutil
import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


def save_checkpoint(state, is_best, check_dir):
    filename = 'latest.pth.tar'
    torch.save(state, os.path.join(check_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(check_dir, filename), 
                        os.path.join(check_dir, 'best.pth.tar'))


def cal_acc(gt_label, pred_result, num):
    acc_sum = 0
    for n in range(num):
        y = []
        pred_y = []
        for i in range(len(gt_label)):
            gt = gt_label[i]
            pred = pred_result[i]
            if gt == n:
                y.append(gt)
                pred_y.append(pred)
        print ('{}: {:4f}'.format(n if n != (num - 1) else 'Unk', accuracy_score(y, pred_y)))
        if n == (num - 1):
            print ('Known Avg Acc: {:4f}'.format(acc_sum / (num - 1)))
        acc_sum += accuracy_score(y, pred_y)
    print ('Avg Acc: {:4f}'.format(acc_sum / num))
    print ('Overall Acc : {:4f}'.format(accuracy_score(gt_label, pred_result)))


def cosine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def to_np(x):
    return x.squeeze().cpu().detach().numpy()


def get_src_centroids(data_loader, model, args):
    feats, labels, probs, preds = get_features(data_loader, model)
    centroids = []
    for i in range(args.class_num - 1):
        data_idx = np.unique(np.argwhere(labels == i))
        feats_i = feats[data_idx].squeeze()

        center_i = np.mean(feats_i, axis=0)
        centroids.append(center_i)

    centroids = np.array(centroids).squeeze()
    return torch.from_numpy(centroids).cuda()


def get_tgt_centroids(data_loader, model, th, src_centroids, args):
    feats, labels, probs, preds = get_features(data_loader, model)
    src_centroids = to_np(src_centroids)
    tgt_dissim = cal_sim(src_centroids, feats, rev=True)
    centroids = []
    for i in range(args.CLASS_NUM - 1):
        class_idx = np.unique(np.argwhere(preds == i))
        easy_idx = np.unique(np.argwhere(tgt_dissim[i, :] <= th))
        data_idx = np.intersect1d(class_idx, easy_idx)
        if len(data_idx) > 1:
            feats_i = feats[data_idx].squeeze()
        else:
            feats_i = np.zeros_like(feats)
            print(i, 'none')
        center_i = np.mean(feats_i, axis=0)
        centroids.append(center_i)

    centroids = np.array(centroids).squeeze()
    return torch.from_numpy(centroids).cuda()


def upd_src_centroids(feats, labels, probs, last_centroids, args):
    new_centroids = []
    feats = to_np(feats)
    labels = to_np(labels)
    last_centroids = to_np(last_centroids)
    probs = F.softmax(probs, dim=1)
    probs = to_np(probs)
    for i in range(args.class_num - 1):
        if np.sum(labels == i) > 0:
            data_idx = np.intersect1d(np.argwhere(labels == i), np.argwhere(probs[:, i] > 0.1))
            new_centroid = np.mean(feats[data_idx], axis=0).reshape(1,-1)
            cs = cosine_similarity(new_centroid, last_centroids[i].reshape(1,-1))[0][0]
            new_centroid = cs * new_centroid + (1 - cs) * last_centroids[i]
        else:
            new_centroid = last_centroids[i]

        new_centroids.append(new_centroid.squeeze())

    new_centroids = np.array(new_centroids)
    return torch.from_numpy(new_centroids).cuda()


def upd_tgt_centroids(feats, probs, last_centroids, src_centroids, args):
    new_centroids = []
    feats = to_np(feats)
    last_centroids = to_np(last_centroids)
    src_centroids = to_np(src_centroids)
    _, ps_labels = probs.max(1, keepdim=True)
    ps_labels = to_np(ps_labels)
    probs = F.softmax(probs, dim=1)
    probs = to_np(probs)
    for i in range(args.CLASS_NUM - 1):
        if np.sum(ps_labels == i) > 0:
            data_idx = np.intersect1d(np.argwhere(ps_labels == i), np.argwhere(probs[:, i] > 0.1))
            new_centroid = np.mean(feats[data_idx], axis=0).reshape(1,-1)

            if last_centroids[i] != np.zeros_like((1, feats.shape[0])):
                cs = cosine_similarity(new_centroid, src_centroids[i].reshape(1,-1))[0][0]
                new_centroid = cs * new_centroid + (1 - cs) * last_centroids[i]
        else:
            new_centroid = last_centroids[i]

        new_centroids.append(new_centroid.squeeze())

    new_centroids = np.array(new_centroids)
    return torch.from_numpy(new_centroids).cuda()


def get_features(data_loader, model):
    model.eval()
    feats, labels = [], []
    probs, preds = [], []
    for batch_idx, batch_data in enumerate(data_loader):
        input, label = batch_data
        input, label = input.cuda(), label.cuda(non_blocking=True)

        feat, prob = model(input)
        prob, pred = prob.max(1, keepdim=True)

        feats.append(feat.cpu().detach().numpy())
        labels.append(label.cpu().detach().numpy())
        probs.append(prob.cpu().detach().numpy())
        preds.append(pred.cpu().detach().numpy())

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    probs = np.concatenate(probs, axis=0)
    preds = np.concatenate(preds, axis=0)
    return feats, labels, probs, preds


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def adjust_learning_rate(optimizer, epoch, args, 
                         step_in_epoch, total_steps_in_epoch):
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    lr = args.lr * cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def cal_sim(x1, x2, metric='cosine'):
    # x = x1.clone()
    if len(x1.shape) != 2:
        x1 = x1.reshape(-1, x1.shape[-1])
    if len(x2.shape) != 2:
        x2 = x2.reshape(-1, x2.shape[-1])

    if metric == 'cosine':
        sim = (F.cosine_similarity(x1, x2) + 1) / 2
    else:
        sim = F.pairwise_distance(x1, x2) / torch.norm(x2, dim=1)
    return sim


def result_log(best_epoch, acc_score, OS_score, all_score, args):
    with open(os.path.join(args.checkpoint, args.log_path), 'a') as f:
        f.write('Task %s\n' % args.task)
        f.write('init_lr %.5f, wd %.5f batch %d\n' % (args.lr, args.weight_decay, args.batch_size))
        f.write('w_s %.5f | w_c %.5f | w_t %.5f\n' % (args.w_s, args.w_c, args.w_t))
        f.write('Best(%d) OS* %.3f OS %.3f ALL %.3f unk %.3f\n' % (best_epoch, acc_score[0], acc_score[1],
                                                                   acc_score[2], acc_score[3]))
        f.write('(OS) OS* %.3f OS %.3f ALL %.3f unk %.3f\n' % (OS_score[0], OS_score[1], OS_score[2], OS_score[3]))
        f.write(
            '(all) OS* %.3f OS %.3f ALL %.3f unk %.3f\n' % (all_score[0], all_score[1], all_score[2], all_score[3]))


# def cal_acc(gt_list, predict_list, num):
#     acc_sum = 0
#     acc_list = {}
#     for n in range(num):
#         y = []
#         pred_y = []
#         for i in range(len(gt_list)):
#             gt = gt_list[i]
#             predict = predict_list[i]
#             if gt == n:
#                 y.append(gt)
#                 pred_y.append(predict)
#         acc = accuracy_score(y, pred_y)
#         print('{}: {:4f}'.format(n if n != (num - 1) else 'Unk', acc))
#         acc_list[n] = acc
#         if n == (num - 1):
#             OS_ = acc_sum * 1.0 / (num - 1)
#             print('Known Avg Acc: {:4f}'.format(OS_))
#             unk = accuracy_score(y, pred_y)
#         acc_sum += accuracy_score(y, pred_y)
#     OS = acc_sum * 1.0 / num
#     all = accuracy_score(gt_list, predict_list)
#     print('Avg Acc: {:4f}'.format(OS))
#     print('Overall Acc : {:4f}\n'.format(all))
#     return OS_, OS, all, unk, acc_list