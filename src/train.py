import argparse
from copy import deepcopy
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from video_dataset import VideoDataset
import alignment_ot
import segmentation_ot
from utils import *
from metrics import ClusteringMetrics, indep_eval_metrics

num_eps = 1e-11


class VideoSSL(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-4, layer_sizes=[64, 128, 40], n_clusters=20, alpha_train=0.3, alpha_eval=0.3,
                 n_ot_train=[50, 1], n_ot_eval=[50, 1], step_size=None, train_eps=0.06, eval_eps=0.01, ub_frames=False, ub_actions=False,
                 lambda_frames_train=0.05, lambda_actions_train=0.05, lambda_frames_eval=0.05, lambda_actions_eval=0.01,
                 temp=0.1, radius_gw=0.04,learn_clusters=True, n_frames=256, rho=0.1, beta=1, exclude_cls=None, visualize=False, activity=None):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_clusters = n_clusters
        self.learn_clusters = learn_clusters
        self.layer_sizes = layer_sizes
        self.exclude_cls = exclude_cls
        self.visualize = visualize

        self.alpha_train = alpha_train
        self.alpha_eval = alpha_eval
        self.n_ot_train = n_ot_train
        self.n_ot_eval = n_ot_eval
        self.step_size = step_size
        self.train_eps = train_eps
        self.eval_eps = eval_eps
        self.radius_gw = radius_gw
        self.ub_frames = ub_frames
        self.ub_actions = ub_actions
        self.lambda_frames_train = lambda_frames_train
        self.lambda_actions_train = lambda_actions_train
        self.lambda_frames_eval = lambda_frames_eval
        self.lambda_actions_eval = lambda_actions_eval

        self.temp = temp
        self.n_frames = n_frames
        self.rho = rho
        self.beta = beta
        # initialize MLP
        layers = [nn.Sequential(nn.Linear(sz, sz1), nn.ReLU()) for sz, sz1 in zip(layer_sizes[:-2], layer_sizes[1:-1])]
        layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]
        self.mlp = nn.Sequential(*layers)

        # initialize cluster centers/codebook
        d = self.layer_sizes[-1]
        self.clusters = nn.parameter.Parameter(data=F.normalize(torch.randn(self.n_clusters, d), dim=-1),
                                               requires_grad=learn_clusters)

        #TODO: Need to replace with evaluation metrics for video alignment
        self.activity = activity
        # initialize evaluation metrics
        self.mof = ClusteringMetrics(metric='mof')
        self.f1 = ClusteringMetrics(metric='f1')
        self.miou = ClusteringMetrics(metric='miou')
        self.save_hyperparameters()
        self.test_cache = []

    def training_step(self, batch, batch_idx):
        # NOTE: The comments below assume batchsize=1 so we dont talk about the batch element dimension
        # features_raw [A 2D Tensor of size=self.n_frames(default 256) x frame embedding dims, representing the input frame embeddings of the sampled frames from a video]
        # mask [A 1D array of Trues/Falses of size=self.n_frames(default 256)]
        # gt [A 1D tensor of action-ids of size=self.n_frames(default 256)]
        # fname [Video's filename]
        # n_subactions [No. of unique action-classes amongst the randomly sampled frames/gt]

        features_raw_X, mask_X, gt_X, fname_X, n_subactions_X = batch[0]
        features_raw_Y, mask_Y, gt_Y, fname_Y, n_subactions_Y = batch[1]

        # MLP's last layer size (D=40 for penn_action)
        D = self.layer_sizes[-1]

        ## Process the features of video X
        # B(batch size), T(no. of frames or timesteps) and _(feature dim/frame embedding size, which is 1024 in penn_action)
        B_X, T_X, _ = features_raw_X.shape
        # Reshape features_raw from 3D(B x T x _) to 2D(B*T x _) tensor, while keeping the feature dim intact
        # Pass it through the MLP: input layer size=_ and output layer size=D
        # Reshape the MLP output from (B*T x D) to (B x T x D)
        # Normalize features along the last dimension(-1), so each feature vector along D and across B and T has unit norm, for stable training
        features_X = F.normalize(
            self.mlp(features_raw_X.reshape(-1, features_raw_X.shape[-1])).reshape(B_X, T_X, D),
            dim=-1
        )

        ## Process the features of video Y
        B_Y, T_Y, _ = features_raw_Y.shape
        features_Y = F.normalize(
            self.mlp(features_raw_Y.reshape(-1, features_raw_Y.shape[-1])).reshape(B_Y, T_Y, D),
            dim=-1
        )

        with torch.no_grad():
            self.clusters.data = F.normalize(self.clusters.data, dim=-1)

        ##FIND SEGMENTATION LOSS FOR X
        codes_segmentation_X = torch.exp(features_X @ self.clusters.T[None, ...] / self.temp)
        codes_segmentation_X = codes_segmentation_X / codes_segmentation_X.sum(dim=-1, keepdim=True)

        with torch.no_grad():  # pseudo-labels from OT
            temp_prior_segmentation_X = segmentation_ot.temporal_prior(T_X, self.n_clusters, self.rho,
                                                                         features_X.device)
            cost_matrix_segmentation_X = 1. - features_X @ self.clusters.T.unsqueeze(0)
            cost_matrix_segmentation_X += temp_prior_segmentation_X
            # print(cost_matrix_segmentation_X.shape)
            #change ub_action to False (reminder)
            opt_codes_segmentation_X, _ = segmentation_ot.segment_ot(cost_matrix_segmentation_X, mask_X,
                                                                         eps=self.train_eps, alpha=self.alpha_train,
                                                                         radius=self.radius_gw,
                                                                         ub_frames=False, ub_actions=True,
                                                                         lambda_frames=self.lambda_frames_train,
                                                                         lambda_actions=self.lambda_actions_train,
                                                                         n_iters=self.n_ot_train,
                                                                         step_size=self.step_size)

        loss_ce_segmentation_X = -(
                    (opt_codes_segmentation_X * torch.log(codes_segmentation_X + num_eps)) * mask_X[..., None]).sum(
            dim=2).mean()
        self.log('train_loss_segmentation_X', loss_ce_segmentation_X)

        ##FIND SEGMENTATION LOSS FOR Y
        codes_segmentation_Y = torch.exp(features_Y @ self.clusters.T[None, ...] / self.temp)
        codes_segmentation_Y = codes_segmentation_Y / codes_segmentation_Y.sum(dim=-1, keepdim=True)

        with torch.no_grad():  # pseudo-labels from OT
            temp_prior_segmentation_Y = segmentation_ot.temporal_prior(T_Y, self.n_clusters, self.rho,
                                                                         features_Y.device)
            cost_matrix_segmentation_Y = 1. - features_Y @ self.clusters.T.unsqueeze(0)
            cost_matrix_segmentation_Y += temp_prior_segmentation_Y
            opt_codes_segmentation_Y, _ = segmentation_ot.segment_ot(cost_matrix_segmentation_Y, mask_Y,
                                                                         eps=self.train_eps, alpha=self.alpha_train,
                                                                         radius=self.radius_gw,
                                                                         ub_frames=False, ub_actions=True,
                                                                         lambda_frames=self.lambda_frames_train,
                                                                         lambda_actions=self.lambda_actions_train,
                                                                         n_iters=self.n_ot_train,
                                                                         step_size=self.step_size)

        loss_ce_segmentation_Y = -(
                    (opt_codes_segmentation_Y * torch.log(codes_segmentation_Y + num_eps)) * mask_Y[..., None]).sum(
            dim=2).mean()
        self.log('train_loss_segmentation_Y', loss_ce_segmentation_Y)

        ##TOTAL SEGMENTATION LOSS
        loss_ce_segmentation = loss_ce_segmentation_X + loss_ce_segmentation_Y

        #######################END SEGMENTATION######################################

        # Eq (6)
        # codes represent a matrix P for each batch element
        # size of a matrix P is (no. of frames in X x no. of frames in Y)
        # P_ij represents the prob. of the frame_i in X being aligned with the frame_j in Y
        codes = torch.exp(features_X @ features_Y.transpose(1, 2) / self.temp)
        codes = codes / codes.sum(dim=-1, keepdim=True)

        # Produce pseudo-labels using ASOT, note that we don't backpropagate through this part
        with torch.no_grad():
            # Calculate the KOT cost matrix from the paragraph above Eq (7)
            # ρR = rho * Temporal prior
            temp_prior = alignment_ot.temporal_prior(T_X, T_Y, self.rho, features_X.device)
            # Cost Matrix Ck from section 4.2, no need to divide by norms since both vectors were previously normalized with F.normalize()
            cost_matrix = 1. - features_X @ features_Y.transpose(1, 2)
            # Ĉk = Ck + ρR
            cost_matrix += temp_prior

            ## Added for virtual frames
            B, N, K = cost_matrix.shape
            dev = cost_matrix.device
            top_row = torch.ones(B, 1, K).to(dev) * 0.5
            cost_matrix = torch.cat((top_row, cost_matrix), dim=1)
            left_column = torch.ones(B, N + 1, 1).to(dev) * 0.5
            cost_matrix = torch.cat((cost_matrix, left_column), dim=2)

            # opt_codes represent a matrix Tb for each batch element
            # size of a matrix Tb is (no. of frames in X x no. of frames in Y)
            # Tb are the (soft) pseudo-labels defined above Eq (7)
            # Tb_ij represents the prob. of the frame_i in X being aligned with the frame_j in Y

            opt_codes, _ = alignment_ot.align_ot(cost_matrix=cost_matrix, mask_X=mask_X, mask_Y=mask_Y,
                                                       eps=self.train_eps, alpha=self.alpha_train,
                                                       radius=self.radius_gw,
                                                       ub_frames=self.ub_frames, ub_actions=self.ub_actions,
                                                       lambda_frames=self.lambda_frames_train,
                                                       lambda_actions=self.lambda_actions_train,
                                                       n_iters=self.n_ot_train, step_size=self.step_size)

        # Eq (7)
        loss_ce_alignment = -((opt_codes * torch.log(codes + num_eps)) * mask_X.unsqueeze(2) * mask_Y.unsqueeze(1)).sum(
            dim=2).mean()
        self.log('train_loss_alignment', loss_ce_alignment)

        # Weighted sum of the segmentation and alignment losses
        # total_loss_ce = (self.beta * loss_ce_segmentation) + loss_ce_alignment
        # print(abc)
        total_loss_ce = loss_ce_segmentation + (self.beta * loss_ce_alignment)
        self.log('train_loss', total_loss_ce)

        return total_loss_ce

    def validation_step(self, batch, batch_idx):
        features_raw_X, mask_X, gt_X, fname_X, n_subactions_X = batch[0]
        features_raw_Y, mask_Y, gt_Y, fname_Y, n_subactions_Y = batch[1]

        D = self.layer_sizes[-1]
        
        B_X, T_X, _ = features_raw_X.shape
        features_X = F.normalize(
            self.mlp(features_raw_X.reshape(-1, features_raw_X.shape[-1])).reshape(B_X, T_X, D),
            dim=-1
        )

        B_Y, T_Y, _ = features_raw_Y.shape
        features_Y = F.normalize(
            self.mlp(features_raw_Y.reshape(-1, features_raw_Y.shape[-1])).reshape(B_Y, T_Y, D),
            dim=-1
        )

        temp_prior_segmentation_X = segmentation_ot.temporal_prior(T_X, self.n_clusters, self.rho,
                                                                     features_X.device)
        cost_matrix_segmentation_X = 1. - features_X @ self.clusters.T.unsqueeze(0)
        cost_matrix_segmentation_X += temp_prior_segmentation_X
        # print(cost_matrix_segmentation_X.shape)
        segmentation_X, _ = segmentation_ot.segment_ot(cost_matrix_segmentation_X, mask_X,
                                                                     eps=self.train_eps, alpha=self.alpha_train,
                                                                     radius=self.radius_gw,
                                                                     ub_frames=False, ub_actions=True,
                                                                     lambda_frames=self.lambda_frames_train,
                                                                     lambda_actions=self.lambda_actions_train,
                                                                     n_iters=self.n_ot_train,
                                                                     step_size=self.step_size)

        temp_prior_segmentation_Y = segmentation_ot.temporal_prior(T_Y, self.n_clusters, self.rho,
                                                                     features_Y.device)
        cost_matrix_segmentation_Y = 1. - features_Y @ self.clusters.T.unsqueeze(0)
        cost_matrix_segmentation_Y += temp_prior_segmentation_Y
        segmentation_Y, _ = segmentation_ot.segment_ot(cost_matrix_segmentation_Y, mask_Y,
                                                                     eps=self.train_eps, alpha=self.alpha_train,
                                                                     radius=self.radius_gw,
                                                                     ub_frames=False, ub_actions=True,
                                                                     lambda_frames=self.lambda_frames_train,
                                                                     lambda_actions=self.lambda_actions_train,
                                                                     n_iters=self.n_ot_train,
                                                                     step_size=self.step_size)
        # log clustering metrics over full epoch


        #TODO: Adjust the evaluation code below for video alignment
        segments_X = segmentation_X.argmax(dim=2)
        self.mof.update(segments_X, gt_X, mask_X)
        self.f1.update(segments_X, gt_X, mask_X)
        self.miou.update(segments_X, gt_X, mask_X)

        segments_Y = segmentation_Y.argmax(dim=2)
        self.mof.update(segments_Y, gt_Y, mask_Y)
        self.f1.update(segments_Y, gt_Y, mask_Y)
        self.miou.update(segments_Y, gt_Y, mask_Y)

        # log clustering metrics per video
        metrics_X = indep_eval_metrics(segments_X, gt_X, mask_X, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        self.log('val_mof_per_X', metrics_X['mof'])
        self.log('val_f1_per_X', metrics_X['f1'])
        self.log('val_miou_per_X', metrics_X['miou'])

        metrics_Y = indep_eval_metrics(segments_Y, gt_Y, mask_Y, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        self.log('val_mof_per_Y', metrics_Y['mof'])
        self.log('val_f1_per_Y', metrics_Y['f1'])
        self.log('val_miou_per_Y', metrics_Y['miou'])

        self.log('val_mof_per', mean([metrics_X['mof'], metrics_Y['mof']]))
        self.log('val_f1_per', mean([metrics_X['f1'], metrics_Y['f1']]))
        self.log('val_miou_per', mean([metrics_X['miou'],metrics_Y['miou']]))
        
        return None
    
    def test_step(self, batch, batch_idx):
        # NOTE: The comments below assume batchsize=1
        # features_raw [A 2D Tensor of size=no. of frames x frame embedding dims, representing the input frame embeddings of a video]
        # mask [A 1D array of Trues of size=no. of frames]
        # gt [A 1D tensor of action-ids of size=no. of frames]
        # fname [Video's filename]
        # n_subactions [No. of unique action-classes amongst the randomly sampled frames/gt]

        features_raw_X, mask_X, gt_X, fname_X, n_subactions_X = batch[0]
        features_raw_Y, mask_Y, gt_Y, fname_Y, n_subactions_Y = batch[1]

        D = self.layer_sizes[-1]

        B_X, T_X, _ = features_raw_X.shape
        features_X = F.normalize(
            self.mlp(features_raw_X.reshape(-1, features_raw_X.shape[-1])).reshape(B_X, T_X, D),
            dim=-1
        )

        B_Y, T_Y, _ = features_raw_Y.shape
        features_Y = F.normalize(
            self.mlp(features_raw_Y.reshape(-1, features_raw_Y.shape[-1])).reshape(B_Y, T_Y, D),
            dim=-1
        )

        temp_prior_segmentation_X = segmentation_ot.temporal_prior(T_X, self.n_clusters, self.rho,
                                                                     features_X.device)
        cost_matrix_segmentation_X = 1. - features_X @ self.clusters.T.unsqueeze(0)
        cost_matrix_segmentation_X += temp_prior_segmentation_X
        # print(cost_matrix_segmentation_X.shape)
        segmentation_X, _ = segmentation_ot.segment_ot(cost_matrix_segmentation_X, mask_X,
                                                           eps=self.train_eps, alpha=self.alpha_train,
                                                           radius=self.radius_gw,
                                                           ub_frames=False, ub_actions=True,
                                                           lambda_frames=self.lambda_frames_train,
                                                           lambda_actions=self.lambda_actions_train,
                                                           n_iters=self.n_ot_train,
                                                           step_size=self.step_size)

        temp_prior_segmentation_Y = segmentation_ot.temporal_prior(T_Y, self.n_clusters, self.rho,
                                                                     features_Y.device)
        cost_matrix_segmentation_Y = 1. - features_Y @ self.clusters.T.unsqueeze(0)
        cost_matrix_segmentation_Y += temp_prior_segmentation_Y
        segmentation_Y, _ = segmentation_ot.segment_ot(cost_matrix_segmentation_Y, mask_Y,
                                                           eps=self.train_eps, alpha=self.alpha_train,
                                                           radius=self.radius_gw,
                                                           ub_frames=False, ub_actions=True,
                                                           lambda_frames=self.lambda_frames_train,
                                                           lambda_actions=self.lambda_actions_train,
                                                           n_iters=self.n_ot_train,
                                                           step_size=self.step_size)

        #TODO: Adjust the evaluation code below for video alignment
        # segments = segmentation.argmax(dim=2)
        #
        # # NOTE: Uncomment this code to store the predictions as a file, this is used on the colab notebook to make visualizations
        # # segments_np = segments.cpu().numpy()
        # # output_dir = "preds"
        # # os.makedirs(output_dir, exist_ok=True)
        # # np.savetxt(os.path.join(output_dir, f'{fname[0]}.txt'), segments_np, fmt='%d', delimiter='\n')
        #
        # self.mof.update(segments, gt, mask)
        # self.f1.update(segments, gt, mask)
        # self.miou.update(segments, gt, mask)
        #
        # # log clustering metrics per video
        # metrics = indep_eval_metrics(segments, gt, mask, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        # self.log('test_mof_per', metrics['mof'])
        # self.log('test_f1_per', metrics['f1'])
        # self.log('test_miou_per', metrics['miou'])

        segments_X = segmentation_X.argmax(dim=2)
        self.mof.update(segments_X, gt_X, mask_X)
        self.f1.update(segments_X, gt_X, mask_X)
        self.miou.update(segments_X, gt_X, mask_X)

        segments_Y = segmentation_Y.argmax(dim=2)
        self.mof.update(segments_Y, gt_Y, mask_Y)
        self.f1.update(segments_Y, gt_Y, mask_Y)
        self.miou.update(segments_Y, gt_Y, mask_Y)

        # log clustering metrics per video
        metrics_X = indep_eval_metrics(segments_X, gt_X, mask_X, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        self.log('val_mof_per_X', metrics_X['mof'])
        self.log('val_f1_per_X', metrics_X['f1'])
        self.log('val_miou_per_X', metrics_X['miou'])

        metrics_Y = indep_eval_metrics(segments_Y, gt_Y, mask_Y, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        self.log('val_mof_per_Y', metrics_Y['mof'])
        self.log('val_f1_per_Y', metrics_Y['f1'])
        self.log('val_miou_per_Y', metrics_Y['miou'])

        self.log('val_mof_per', mean([metrics_X['mof'], metrics_Y['mof']]))
        self.log('val_f1_per', mean([metrics_X['f1'], metrics_Y['f1']]))
        self.log('val_miou_per', mean([metrics_X['miou'],metrics_Y['miou']]))
        # cache videos for plotting
        self.test_cache.append([metrics_X['mof'], segments_X, gt_X, mask_X, fname_X])
        self.test_cache.append([metrics_Y['mof'], segments_Y, gt_Y, mask_Y, fname_Y])

        return None
    
    def on_validation_epoch_end(self):
        #TODO: Adjust the evaluation code below for video alignment
        mof, pred_to_gt = self.mof.compute(exclude_cls=self.exclude_cls)
        f1, _ = self.f1.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        miou, _ = self.miou.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        self.log('val_mof_full', mof)
        self.log('val_f1_full', f1)
        self.log('val_miou_full', miou)
        self.mof.reset()
        self.f1.reset()
        self.miou.reset()

    def on_test_epoch_end(self):
        #TODO: Adjust the evaluation code below for video alignment
        mof, pred_to_gt = self.mof.compute(exclude_cls=self.exclude_cls)
        f1, _ = self.f1.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        miou, _  = self.miou.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        self.log('test_mof_full', mof)
        self.log('test_f1_full', f1)
        self.log('test_miou_full', miou)

        if self.visualize:
            
            # Make a dir to store visualizations
            output_dir = f'{self.activity}'
            os.makedirs(output_dir, exist_ok=True)

            for i, (mof, pred, gt, mask, fname) in enumerate(self.test_cache):
                self.test_cache[i][0] = indep_eval_metrics(pred, gt, mask, ['mof'], exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)['mof']
            self.test_cache = sorted(self.test_cache, key=lambda x: x[0], reverse=True)

            for i, (mof, pred, gt, mask, fname) in enumerate(self.test_cache):
                fig = plot_segmentation_gt(gt, pred, mask, exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt,
                                           gt_uniq=np.unique(self.mof.gt_labels), name=f'{fname[0]}')
                fig_path = os.path.join(output_dir, f"test_segment_{i}.png")
                fig.savefig(fig_path)
                plt.close(fig)

        self.test_cache = []
        self.mof.reset()
        self.f1.reset()
        self.miou.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train representation learning pipeline")

    #TODO: Change args to only keep what we need and add any new args that we might need
    # FUGW OT segmentation parameters
    parser.add_argument('--alpha-train', '-at', type=float, default=0.3, help='weighting of KOT term on frame features in OT')
    parser.add_argument('--alpha-eval', '-ae', type=float, default=0.6, help='weighting of KOT term on frame features in OT')
    parser.add_argument('--ub-frames', '-uf', action='store_true',
                        help='relaxes balanced assignment assumption over frames, i.e., each frame is assigned')
    parser.add_argument('--ub-actions', '-ua', action='store_true',
                        help='relaxes balanced assignment assumption over actions, i.e., each action is uniformly represented in a video')
    parser.add_argument('--lambda-frames-train', '-lft', type=float, default=0.05, help='penalty on balanced frames assumption for training')
    parser.add_argument('--lambda-actions-train', '-lat', type=float, default=0.05, help='penalty on balanced actions assumption for training')
    parser.add_argument('--lambda-frames-eval', '-lfe', type=float, default=0.05, help='penalty on balanced frames assumption for test')
    parser.add_argument('--lambda-actions-eval', '-lae', type=float, default=0.01, help='penalty on balanced actions assumption for test')
    parser.add_argument('--eps-train', '-et', type=float, default=0.07, help='entropy regularization for OT during training')
    parser.add_argument('--eps-eval', '-ee', type=float, default=0.04, help='entropy regularization for OT during val/test')
    parser.add_argument('--radius-gw', '-r', type=float, default=0.04, help='Radius parameter for GW structure loss')
    parser.add_argument('--n-ot-train', '-nt', type=int, nargs='+', default=[25, 1], help='number of outer and inner iterations for ASOT solver (train)')
    parser.add_argument('--n-ot-eval', '-no', type=int, nargs='+', default=[25, 1], help='number of outer and inner iterations for ASOT solver (eval)')
    parser.add_argument('--step-size', '-ss', type=float, default=None,
                        help='Step size/learning rate for ASOT solver. Worth setting manually if ub-frames && ub-actions')

    # dataset params
    ## TODO: Not using this arg since we hardcoded it, add it back in later
    parser.add_argument('--base-path', '-p', type=str, default='/home/users/u6567085/data', help='base directory for dataset')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='dataset to use for training/eval (Breakfast, YTI, FSeval, FS, desktop_assembly)')
    parser.add_argument('--activity', '-ac', type=str, nargs='+', required=True, help='activity classes to select for dataset')
    parser.add_argument('--exclude', '-x', type=int, default=None, help='classes to exclude from evaluation. use -1 for YTI')
    parser.add_argument('--n-frames', '-f', type=int, default=256, help='number of frames sampled per video for train/val')
    parser.add_argument('--std-feats', '-s', action='store_true', help='standardize features per video during preprocessing')
    
    # representation learning params
    parser.add_argument('--n-epochs', '-ne', type=int, default=15, help='number of epochs for training')
    parser.add_argument('--batch-size', '-bs', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', '-wd', type=float, default=1e-4, help='weight decay for optimizer')
    parser.add_argument('--layers', '-ls', default=[64, 128, 40], nargs='+', type=int, help='layer sizes for MLP (in, hidden, ..., out)')
    parser.add_argument('--rho', type=float, default=0.1, help='Factor for global structure weighting term')
    parser.add_argument('--n-clusters', '-c', type=int, default=8, help='number of actions/clusters')
    parser.add_argument('--beta', '-b', type=float, default=1,
                        help='the weight used when combining alignment and segmentation losses')

    # system/logging params
    parser.add_argument('--val-freq', '-vf', type=int, default=5, help='validation epoch frequency (epochs)')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='gpu id to use')
    parser.add_argument('--visualize', '-v', action='store_true', help='generate visualizations during logging')
    parser.add_argument('--seed', type=int, default=0, help='Random seed initialization')
    parser.add_argument('--ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--eval', action='store_true', help='run evaluation on test set only')
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    
    #TODO: Modify VideoDataset, DataLoader and create a new file structure to read 2 videos and their labels
    # Set the paths for the data directory, its structure is mentioned in the README 
    data_val = VideoDataset('../data', args.dataset, args.n_frames, standardise=args.std_feats, random=False, action_class=args.activity)
    data_train = VideoDataset('../data', args.dataset, args.n_frames, standardise=args.std_feats, random=True, action_class=args.activity)
    # Only difference is passing n_frames=None
    data_test = VideoDataset('../data', args.dataset, None, standardise=args.std_feats, random=False, action_class=args.activity)
    val_loader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False)

    if args.ckpt is not None:
        ssl = VideoSSL.load_from_checkpoint(args.ckpt)
    
    #TODO: Initialize using only the arguments we will need
    else:
        ssl = VideoSSL(layer_sizes=args.layers, n_clusters=args.n_clusters, alpha_train=args.alpha_train, alpha_eval=args.alpha_eval,
                       ub_frames=args.ub_frames, ub_actions=args.ub_actions, lambda_frames_train=args.lambda_frames_train, lambda_frames_eval=args.lambda_frames_eval,
                       lambda_actions_train=args.lambda_actions_train, lambda_actions_eval=args.lambda_actions_eval, step_size=args.step_size,
                       train_eps=args.eps_train, eval_eps=args.eps_eval, radius_gw=args.radius_gw, n_ot_train=args.n_ot_train, n_ot_eval=args.n_ot_eval,
                       n_frames=args.n_frames, lr=args.learning_rate, weight_decay=args.weight_decay, rho=args.rho,  beta=args.beta, exclude_cls=args.exclude, visualize=args.visualize, activity=args.activity)
    
    # TODO: Added num_sanity_val_steps=0 to disable the initial validation step while I fix the training_step, remove it later
    trainer = pl.Trainer(devices=1, accelerator='gpu', check_val_every_n_epoch=args.val_freq, max_epochs=args.n_epochs, log_every_n_steps=50, logger=None, num_sanity_val_steps=0)

    if not args.eval:
        # TODO: Uncomment this once the evaluation metrics are fixed
        trainer.validate(ssl, val_loader)
        trainer.fit(ssl, train_loader, val_loader)
    trainer.test(ssl, dataloaders=test_loader)
