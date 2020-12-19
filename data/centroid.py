from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from utils import *


class Centroids(object):
    def __init__(self, class_num, dim, use_cuda):
        self.class_num = class_num
        self.src_ctrs = t.ones((class_num, dim))
        self.tgt_ctrs = t.ones((class_num, dim))
        self.src_ctrs *= 1e-10
        self.tgt_ctrs *= 1e-10
        if use_cuda:
            self.src_ctrs = self.src_ctrs.cuda()
            self.tgt_ctrs = self.tgt_ctrs.cuda()
            

    def get_centroids(self, domain=None, cid=None):
        if domain == 'source':
            return self.src_ctrs if cid is None else self.src_ctrs[cid, :]
        elif domain == 'target':
            return self.tgt_ctrs if cid is None else self.tgt_ctrs[cid, :]
        else:
            return self.src_ctrs, self.tgt_ctrs

    @torch.no_grad()
    def update(self, feat_s, pred_s, label_s, feat_t, pred_t):
        self.upd_src_centroids(feat_s, pred_s, label_s)
        self.upd_tgt_centroids(feat_t, pred_t)
        
    @torch.no_grad()
    def upd_src_centroids(self, feats, probs, labels):
        # feats = to_np(feats)
        labels = to_np(labels)
        # last_centroids = to_np(self.src_ctrs)
        probs = to_np(F.softmax(probs, dim=1))

        for i in range(self.class_num - 1):
            if np.sum(labels == i) > 1:
                last_centroid = self.src_ctrs[i, :]
                data_idx = np.argwhere(labels == i)
                new_centroid = t.mean(feats[data_idx, :], 0).squeeze()
                cs = cal_sim(new_centroid, last_centroid)
                # print(cs)
                new_centroid = cs * new_centroid + (1 - cs) * last_centroid
                self.src_ctrs[i, :] = new_centroid

    @torch.no_grad()
    def upd_tgt_centroids(self, feats, probs):
        # feats = to_np(feats)
        # last_centroids = to_np(self.tgt_ctrs)
        # src_centroids = to_np(self.src_ctrs)
        _, pseudo_label = probs.max(1, keepdim=True)
        pseudo_label = to_np(pseudo_label)
        probs = to_np(F.softmax(probs, dim=1))

        for i in range(self.class_num):
            if np.sum(pseudo_label == i) > 1:
                data_idx = np.argwhere(pseudo_label == i)
                new_centroid = t.mean(feats[data_idx, :], 0).squeeze()
                last_centroid = self.tgt_ctrs[i, :]
                # if last_centroids[i] != np.zeros_like((1, feats.shape[0])):
                cs = cal_sim(new_centroid, self.src_ctrs[i, :])
                # print(cs)
                new_centroid = cs * new_centroid + (1 - cs) * last_centroid
                self.tgt_ctrs[i, :] = new_centroid


def crit_intra(feats, y, centers, lambd=1e-3):
    class_num = len(centers)
    batch_size = y.shape[0]

    expanded_centers = centers.expand(batch_size, -1, -1)
    expanded_feats = feats.expand(class_num, -1, -1).transpose(1, 0)
    # distance_centers = (expanded_feats - expanded_centers).pow(2).sum(dim=-1)
    distance_centers = cal_sim(expanded_feats, expanded_centers)
    distance_centers = distance_centers.reshape(batch_size, class_num)

    intra_distances = distance_centers.gather(1, y.unsqueeze(1))
    # intra_distances = distances_same.sum()
    inter_distances = distance_centers.sum(dim=-1) - intra_distances

    epsilon = 1e-6
    loss = (1 / 2.0 / batch_size / class_num) * intra_distances / \
           (inter_distances + epsilon)
    loss = loss.sum()
    loss *= lambd
    return loss


def crit_inter(center1, center2, lambd=1e-3):
    # dists = F.pairwise_distance(center1, center2)
    # loss = t.mean(dists)

    # dists = cal_cossim(center1.cpu().numpy(), center2.cpu().numpy())
    dists = cal_sim(center1, center2)
    loss = 0
    for i in range(center1.shape[0]):
        loss += dists[i]#[i]
    loss /= center1.shape[0]
    loss *= lambd
    return loss, dists


def crit_contrast(feats, probs, s_ctds, t_ctds, lambd=1e-3):
    batch_num = feats.shape[0]
    class_num = s_ctds.shape[0]
    probs = F.softmax(probs, dim=-1)
    max_probs, preds = probs.max(1, keepdim=True)
    # print(probs.shape, max_probs.shape)
    select_index = t.nonzero(max_probs.squeeze() >= 0.3).squeeze(1)
    select_index = select_index.cpu().tolist()

    # todo: calculate margins
    # dist_ctds = cal_cossim(to_np(s_ctds), to_np(t_ctds))
    dist_ctds = cal_sim(s_ctds, t_ctds)
    # print('dist_ctds', dist_ctds.shape)

    M = np.ones(class_num)
    for i in range(class_num):
        # M[i] = np.sum(dist_ctds[i, :]) - dist_ctds[i, i]
        M[i] = dist_ctds.mean() - dist_ctds[i]
        M[i] /= class_num - 1
    # print('M', M)

    # todo: calculate D_k between known samples to its source centroid &
    # todo: calculate D_u distances between unknown samples to all source centroids
    D_k, n_k = 0, 1e-5
    D_u, n_u = 0, 1e-5
    for i in select_index:
        class_id = preds[i][0]
        if class_id < class_num:
            # D_k += F.pairwise_distance(feats[i, :], s_ctds[class_id]).squeeze()
            # print(feats.shape, i)
            D_k += cal_sim(feats[i, :], s_ctds[class_id, :])
            # print('D_k', D_k)
            n_k += 1
        else:
            # todo: judge if unknown sample in the radius region of known centroid
            rp_feats = feats[i, :].unsqueeze(0).repeat(class_num, 1)

            # dist_known = F.pairwise_distance(rp_feats, s_ctds)
            dist_known = cal_sim(rp_feats, s_ctds)
            # print('dist_known', len(dist_known), dist_known)

            M_mean = M.mean()
            outliers = dist_known < M_mean
            dist_margin = (dist_known - M_mean) * outliers.float()
            D_u += dist_margin.sum()

    loss = D_k / n_k  # - D_u / n_u
    return loss.mean() * lambd
