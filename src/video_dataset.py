import os
import os.path as path

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_action_name(fname, dataset):
    if dataset == 'Breakfast':
        return fname.split('_')[-1]
    if dataset == 'YTI':  # ignores _idt files in groundTruth automatically
        return '_'.join(fname.split('_')[:-1])
    if dataset == 'FS':  # only one activity class
        return ''
    if dataset == 'desktop_assembly':  # only one activity class
        return ''
    if dataset == 'penn_action':  # only one activity class (TODO:for now)
        return ''
    raise ValueError(f'{dataset} is not a valid dataset!')


class VideoDataset(Dataset):
    def __init__(self, root_dir: str, dataset, n_frames, standardise=True, split: str = None, random=True, n_videos=None, action_class=['all']):
        self.root_dir = root_dir
        self.dataset = dataset
        if self.dataset == 'FSeval':
            self.dataset = 'FS'
            granularity = 'eval'
        else:
            granularity = None
        self.data_dir = path.join(root_dir, self.dataset)

        # penn_action files don't have a '-' or '_' in the name
        if self.dataset == 'penn_action':
            self.video_fnames = sorted([fname for fname in os.listdir(path.join(self.data_dir, 'groundTruth'))])
        # Store a sorted list of all the groundTruth filenames that have a '-' or '_' anywhere in the name
        else:
            self.video_fnames = sorted([fname for fname in os.listdir(path.join(self.data_dir, 'groundTruth'))
                                    if len(fname.split('_')) > 1 or len(fname.split('-')) > 1])

        if self.dataset in ['FS', 'desktop_assembly', 'penn_action']:
            action_class = ''
        if action_class != ['all']:
            if type(action_class) is list:
                self.video_fnames = [fname for fname in self.video_fnames if parse_action_name(fname, self.dataset) in action_class]
            else:
                self.video_fnames = [fname for fname in self.video_fnames if parse_action_name(fname, self.dataset) == action_class]
        if n_videos is not None:
            # inds = np.random.permutation(len(self.video_fnames))[:n_videos]
            # self.video_fnames = sorted([self.video_fnames[ind] for ind in inds])
            self.video_fnames = self.video_fnames[::int(len(self.video_fnames) / n_videos)]
        def prep(x):
            i, nm = x.rstrip().split(' ')
            return nm, int(i)
        if granularity is None:  # granularity applies only to 50Salads
            # Apply prep() to each line in the file to convert each line into a tuple (action-class, action-id), store a list of tuples
            action_mapping = list(map(prep, open(path.join(self.data_dir, 'mapping/mapping.txt'))))
        else:
            action_mapping = list(map(prep, open(path.join(self.data_dir, f'mapping/mapping{granularity}.txt'))))
        self.action_mapping = dict(action_mapping)
        self.n_subactions = len(set(self.action_mapping.keys()))
        self.n_frames = n_frames
        self.standardise = standardise
        self.random = random

    def __len__(self):
        return len(self.video_fnames)
    
    def __getitem__(self, idx):
        # Get the filename for video X
        video_fname_X = self.video_fnames[idx]
        
        # Randomly select another video index ensuring its different from idx
        other_idx = (idx + np.random.randint(1, len(self.video_fnames))) % len(self.video_fnames)
        # Get the filename for video Y
        video_fname_Y = self.video_fnames[other_idx]

        # Load the required data for each video
        features_X, mask_X, gt_X, video_fname_X, unique_actions_X = self._load_video_data(video_fname_X)
        features_Y, mask_Y, gt_Y, video_fname_Y, unique_actions_Y = self._load_video_data(video_fname_Y)

        # Return it for the batch
        return (features_X, mask_X, gt_X, video_fname_X, unique_actions_X), (features_Y, mask_Y, gt_Y, video_fname_Y, unique_actions_Y)

    def _load_video_data(self, video_fname):
        # Load this vid's groundTruth file, convert it into a list of action-classes
        gt = [line.rstrip() for line in open(path.join(self.data_dir, 'groundTruth', video_fname))]
        
        # __getitem__ is supposed to return the features and labels corresponding to the frames in a video, 
        # but sometimes we don't want to use all of the frames in the video, so we sample some amount of frames
        # calling this function determines exactly what frames from the video we will be using
        inds, mask = self._partition_and_sample(self.n_frames, len(gt))

        # We will use inds, to store the features and gt for only the sampled frames
        # Converts the list of action-classes into a PyTorch tensor of action-ids, filtered by inds
        # Now ground truth is represented by a tensor of longs
        gt = torch.Tensor([self.action_mapping[gt[ind]] for ind in inds]).long()

        # Load features
        action = parse_action_name(video_fname, self.dataset)
        feat_fname = path.join(self.data_dir, 'features', action, video_fname)
        try:
            features = np.loadtxt(feat_fname + '.txt')[inds, :]
        except:
            features = np.load(feat_fname + '.npy')[inds, :]

        if self.standardise:  # normalize features
            zmask = np.ones(features.shape[0], dtype=bool)
            for rdx, row in enumerate(features):
                if np.sum(row) == 0:
                    zmask[rdx] = False
            z = features[zmask] - np.mean(features[zmask], axis=0)
            z = z / np.std(features[zmask], axis=0)
            features = np.zeros(features.shape)
            features[zmask] = z
            features = np.nan_to_num(features)
            features /= np.sqrt(features.shape[1])
        
        features = torch.from_numpy(features).float()
        return features, mask, gt, video_fname, gt.unique().shape[0]
    
    # Receives: how many frames are there in the video(n_frames) and how many frames would we like to sample from the video(n_samples)
    #           and should we perform random sampling(self.random)
    # Returns: the indices of the frames to use(indices), which of the frames are unique???TODO???(True) and which are padding(False) (mask).
    # NOTE: For all conditions except first, len(indices)==len(mask)==n_samples.
    def _partition_and_sample(self, n_samples, n_frames):
        # This condition is used for test, where n_samples=None, meaning use all the frames in the video
        if n_samples is None:
            # Values are [0 to n_frames-1]
            indices = np.arange(n_frames)
            # All values are True
            mask = np.full(n_frames, 1, dtype=bool)
        
        # These conditions are used for train/val, where n_samples=args.n_frames(default 256)
        # If no. of samples required < no. of frames in the video
        elif n_samples < n_frames:
            # If random sampling, indices will be filled with n_samples random indices from [0 to n_frames-1]  
            if self.random:
                boundaries = np.linspace(0, n_frames-1, n_samples+1).astype(int)
                indices = np.random.randint(low=boundaries[:-1], high=boundaries[1:])
            # Else, indices will be filled with n_samples evenly spaced indices from [0 to n_frames-1]
            else:
                indices = np.linspace(0, n_frames-1, n_samples).astype(int)
            # All values are True
            mask = np.full(n_samples, 1, dtype=bool)
        
        # If no. of samples required >= no. of frames in the video
        else:
            # First n_frames values are [0 to n_frames-1] and remaining values(if any) are n_frames-1
            indices = np.concatenate((np.arange(n_frames), np.full(n_samples - n_frames, n_frames - 1)))
            # First n_frames values are True and remaining values(if any) are False
            mask = np.concatenate((np.full(n_frames, 1, dtype=bool), np.zeros(n_samples - n_frames, dtype=bool)))
        
        return indices, mask
