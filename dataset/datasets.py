"""
File containing the function to load all the frame datasets.
"""

#Standard imports
import os

#Local imports
from util.dataset import load_classes
from dataset.frame import ActionSpotDataset, ActionSpotDatasetJoint

#Constants
STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 4
STRIDE_SNBA = 4
OVERLAP = 0.9
OVERLAP_SN = 0.5
OVERLAP_SNBA = 0.9

def get_datasets(args, label_pad_idx, n_class):
    # Set dataset arguments
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    dataset_len = args.epoch_num_frames // args.clip_len
    stride = STRIDE
    overlap = OVERLAP
    if args.dataset == 'soccernet':
        stride = STRIDE_SN
        overlap = OVERLAP_SN
    elif args.dataset == 'soccernetball':
        stride = STRIDE_SNB
    elif args.dataset == 'soccernetballanticipation':
        stride = STRIDE_SNBA
        overlap = OVERLAP_SNBA

    dataset_kwargs = {
        'stride': stride, 'overlap': overlap, 'radi_smoothing': args.radi_smoothing,
        'dataset': args.dataset, 'obs_perc': args.obs_perc,
        'pred_perc': args.pred_perc, 'n_query': args.n_query, "excluded_classes": args.excluded_classes,
        'anticipate_background': args.anticipate_background, 'use_actionness': args.actionness,
        'use_anchors': args.use_anchors, 'cheating_dataset': args.cheating_dataset,
        'cheating_range': args.cheating_range,
        'use_optical_flow': args.use_optical_flow,
        'flow_subdir': args.flow_subdir,
        'flow_fps': args.flow_fps,
    }

    # -----------------------------------------------------------------------
    # Optional: re-partition the 720p/train folder into a local train / val
    # split without moving any files.  Controlled by args.train_local_split.
    # -----------------------------------------------------------------------
    use_local_split = (
        getattr(args, 'train_local_split', None) is not None
        and args.dataset == 'soccernetballanticipation'
    )

    # Create training dataset
    print('Dataset size:', dataset_len)
    if use_local_split:
        import json as _json
        with open(os.path.join('data', args.dataset, 'train.json')) as _f:
            _train_meta = _json.load(_f)
        # train.json is a list with one entry {"video": "train", "num_clips": N, ...}
        total_clips = int(_train_meta[0]['num_clips'])
        split_idx = int(total_clips * args.train_local_split)
        print(f'Local split: using clips 0–{split_idx} for training, '
              f'{split_idx}–{total_clips} for local validation '
              f'(train_local_split={args.train_local_split})')
        train_data = ActionSpotDataset(
            classes, os.path.join('data', args.dataset, 'train.json'),
            args.frame_dir, args.store_dir, args.store_mode,
            args.clip_len, dataset_len,
            label_pad_idx, n_class,
            clip_idx_range=(0, split_idx),
            **dataset_kwargs)
        train_data.print_info()
        val_data = ActionSpotDataset(
            classes, os.path.join('data', args.dataset, 'train.json'),
            args.frame_dir, args.store_dir, args.store_mode,
            args.clip_len, dataset_len // 4,
            label_pad_idx, n_class,
            clip_idx_range=(split_idx, total_clips),
            **dataset_kwargs)
        val_data.print_info()
    else:
        if args.cheating_dataset:
            train_data = ActionSpotDataset(
                classes, os.path.join('data', args.dataset, 'train.json'),
                args.frame_dir, args.store_dir, args.store_mode, 
                args.clip_len, dataset_len,
                label_pad_idx, n_class, **dataset_kwargs)
            print("Cheating training dataset")
        else:
            train_data = ActionSpotDataset(
                classes, os.path.join('data', args.dataset, 'train.json'),
                args.frame_dir, args.store_dir, args.store_mode, 
                args.clip_len, dataset_len,
                label_pad_idx, n_class, **dataset_kwargs)
        train_data.print_info()

        # Create validation dataset
        if args.cheating_dataset:
            val_data = ActionSpotDataset(
                classes, os.path.join('data', args.dataset, 'val.json'),
                args.frame_dir, args.store_dir, args.store_mode,
                args.clip_len, dataset_len // 4,
                label_pad_idx, n_class, **dataset_kwargs)
            print("Cheating validation dataset")
        else:
            val_data = ActionSpotDataset(
                classes, os.path.join('data', args.dataset, 'val.json'),
                args.frame_dir, args.store_dir, args.store_mode,
                args.clip_len, dataset_len // 4,
                label_pad_idx, n_class, **dataset_kwargs)
        val_data.print_info()

    val_data_frames = None     
        
    #In case of using jointtrain, datasets with additional data
    jointtrain_classes = None
    if args.jointtrain != None:

        stride_jointtrain = STRIDE
        overlap_jointtrain = OVERLAP
        if args.jointtrain['dataset'] == 'soccernet':
            stride_jointtrain = STRIDE_SNB
            overlap_jointtrain = OVERLAP_SN
        elif args.dataset == 'soccernetball':
            stride_jointtrain = STRIDE_SNB
        elif args.dataset == 'soccernetballanticipation':
            stride_jointtrain = STRIDE_SNBA
            overlap_jointtrain = OVERLAP_SNBA

        dataset_jointtrain_kwargs = {
            'stride': stride_jointtrain, 'overlap': overlap_jointtrain, 'radi_smoothing': args.radi_smoothing,
            'dataset': args.jointtrain['dataset'], 'obs_perc': args.obs_perc,
            'pred_perc': args.pred_perc, 'n_query': args.n_query, "excluded_classes": args.excluded_classes,
            'anticipate_background': args.anticipate_background, 'use_actionness': args.actionness,
            'use_anchors': args.use_anchors, 'cheating_dataset': args.cheating_dataset,
            'cheating_range': args.cheating_range,
        }

        jointtrain_classes = load_classes(os.path.join('data', args.jointtrain['dataset'], 'class.txt'))
        n_class_jointtrain = len(jointtrain_classes)

        # Create training joint-train dataset
        if args.cheating_dataset:
            jointtrain_train_data = ActionSpotDataset(
                jointtrain_classes, os.path.join('data', args.jointtrain['dataset'], 'train.json'),
                args.jointtrain['frame_dir'], args.jointtrain['store_dir'], args.store_mode,
                args.clip_len, dataset_len,
                label_pad_idx, n_class_jointtrain, **dataset_jointtrain_kwargs)
            print("Cheating training joint-training dataset")
        else:
            jointtrain_train_data = ActionSpotDataset(
                jointtrain_classes, os.path.join('data', args.jointtrain['dataset'], 'train.json'),
                args.jointtrain['frame_dir'], args.jointtrain['store_dir'], args.store_mode,
                args.clip_len, dataset_len,
                label_pad_idx, n_class_jointtrain, **dataset_jointtrain_kwargs)
        jointtrain_train_data.print_info()

        # Create validation joint-train dataset
        if args.cheating_dataset:
            jointtrain_val_data = ActionSpotDataset(
                jointtrain_classes, os.path.join('data', args.jointtrain['dataset'], 'val.json'),
                args.jointtrain['frame_dir'], args.jointtrain['store_dir'], args.store_mode,
                args.clip_len, dataset_len // 4,
                label_pad_idx, n_class_jointtrain, **dataset_jointtrain_kwargs)
            print("Cheating validation joint-training dataset")
        else:
            jointtrain_val_data = ActionSpotDataset(
                jointtrain_classes, os.path.join('data', args.jointtrain['dataset'], 'val.json'),
                args.jointtrain['frame_dir'], args.jointtrain['store_dir'], args.store_mode,
                args.clip_len, dataset_len // 4,
                label_pad_idx, n_class_jointtrain, **dataset_jointtrain_kwargs)
        jointtrain_val_data.print_info()

        # Create the joint datasets used for joint training
        train_data = ActionSpotDatasetJoint(train_data, jointtrain_train_data)
        val_data = ActionSpotDatasetJoint(val_data, jointtrain_val_data)
        
    return classes, jointtrain_classes, train_data, val_data, val_data_frames