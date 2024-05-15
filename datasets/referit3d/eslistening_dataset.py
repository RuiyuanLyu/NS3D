"""
From ReferIt3D (https://github.com/referit3d/referit3d)

The MIT License (MIT)
Originally created at 5/25/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Ahmed (@gmail.com)
"""

import numpy as np
from torch.utils.data import Dataset
from functools import partial
import warnings
import multiprocessing as mp
from torch.utils.data import DataLoader
import torch
import os
# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from datasets.referit3d.referit3d_reader import decode_stimulus_string


def max_io_workers():
    """ number of available cores -1."""
    n = max(mp.cpu_count() - 1, 1)
    print('Using {} cores for I/O.'.format(n))
    return n


def dataset_to_dataloader(dataset, split, batch_size, n_workers, pin_memory=False, seed=None):
    """
    :param dataset:
    :param split:
    :param batch_size:
    :param n_workers:
    :param pin_memory:
    :param seed:
    :return:
    """
    batch_size_multiplier = 1 if split == 'train' else 2
    b_size = int(batch_size_multiplier * batch_size)

    drop_last = False
    if split == 'train' and len(dataset) % b_size == 1:
        print('dropping last batch during training')
        drop_last = True

    shuffle = split == 'train'

    worker_init_fn = lambda x: np.random.seed(seed)
    if split == 'test':
        if type(seed) is not int:
            warnings.warn('Test split is not seeded in a deterministic manner.')

    data_loader = DataLoader(dataset,
                             batch_size=b_size,
                             num_workers=n_workers,
                             shuffle=shuffle,
                             drop_last=drop_last,
                             pin_memory=pin_memory,
                             worker_init_fn=worker_init_fn)
    return data_loader


# def sample_scan_object(object, n_points):
#     sample = object.sample(n_samples=n_points)
#     return np.concatenate([sample['xyz'], sample['color']], axis=1)


def pad_samples(samples, max_context_size, padding_value=1):
    n_pad = max_context_size - len(samples)

    if n_pad > 0:
        shape = (max_context_size, samples.shape[1], samples.shape[2])
        temp = np.ones(shape, dtype=samples.dtype) * padding_value
        temp[:samples.shape[0], :samples.shape[1]] = samples
        samples = temp

    return samples


def check_segmented_object_order(scans):
    """ check all scan objects have the three_d_objects sorted by id
    :param scans: (dict)
    """
    for scan_id, scan in scans.items():
        idx = scan.three_d_objects[0].object_id
        for o in scan.three_d_objects:
            if not (o.object_id == idx):
                print('Check failed for {}'.format(scan_id))
                return False
            idx += 1
    return True


def objects_bboxes(context):
    b_boxes = []
    for o in context:
        bbox = o.get_bbox(axis_aligned=True)

        # Get the centre
        cx, cy, cz = bbox.cx, bbox.cy, bbox.cz

        # Get the scale
        lx, ly, lz = bbox.lx, bbox.ly, bbox.lz

        b_boxes.append([cx, cy, cz, lx, ly, lz])

    return np.array(b_boxes).reshape((len(context), 6))


def instance_labels_of_context(context, max_context_size, label_to_idx=None, add_padding=True):
    """
    :param context: a list of the objects
    :return:
    """
    instance_labels = [i.instance_label for i in context]

    if add_padding:
        n_pad = max_context_size - len(context)
        instance_labels.extend(['pad'] * n_pad)

    if label_to_idx is not None:
        instance_labels = np.array([label_to_idx[x] for x in instance_labels])

    return instance_labels


def mean_rgb_unit_norm_transform(segmented_objects, mean_rgb, unit_norm, epsilon_dist=10e-6, inplace=True):
    """
    :param segmented_objects: K x n_points x 6, K point-clouds with color.
    :param mean_rgb:
    :param unit_norm:
    :param epsilon_dist: if max-dist is less than this, we apply not scaling in unit-sphere.
    :param inplace: it False, the transformation is applied in a copy of the segmented_objects.
    :return:
    """
    if not inplace:
        segmented_objects = segmented_objects.copy()

    # adjust rgb
    segmented_objects[:, :, 3:6] -= np.expand_dims(mean_rgb, 0)

    # center xyz
    if unit_norm:
        xyz = segmented_objects[:, :, :3]
        mean_center = xyz.mean(axis=1)
        xyz -= np.expand_dims(mean_center, 1)
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=-1)), -1)
        max_dist[max_dist < epsilon_dist] = 1  # take care of tiny point-clouds, i.e., padding
        xyz /= np.expand_dims(np.expand_dims(max_dist, -1), -1)
        segmented_objects[:, :, :3] = xyz
    return segmented_objects


from utils.utils_read import read_annotation_pickles
import json


class ESListeningDataset(Dataset):
    def __init__(self, es_info_file, vg_raw_data_file, processed_scan_dir, vocab, object_transformation=None,):
        super().__init__()
        self.es_info = read_annotation_pickles(es_info_file)
        self.class_to_idx = np.load(es_info_file, allow_pickle=True)["metainfo"]["categories"]
        # self.es_info[scene_id] = {
        #     "bboxes": bboxes,
        #     "object_ids": object_ids,
        #     "object_types": object_types,
        #     "visible_view_object_dict": visible_view_object_dict,
        #     "extrinsics_c2w": extrinsics_c2w,
        #     "axis_align_matrix": axis_align_matrix,
        #     "intrinsics": intrinsics,
        #     "depth_intrinsics": depth_intrinsics,
        #     "image_paths": image_paths,
        # }
        self.vg_raw_data = json.load(open(vg_raw_data_file, 'r'))
        self.scan_ids = list(self.es_info.keys())
        self.num_scans = len(self.scan_ids)
        self.num_points = 1024 # number of points to sample from each object
        # 需要维护两份数据，一份是关于bbox和pcd的，一份是关于vg txt的
        # vg txt的一个dict的例子：{'scan_id': 'scene0000_00', 'target_id': ['30'], 'distractor_ids': [], 'text': 'The X is used for sitting at the table.Please find the X.', 'target': ['stool'], 'anchors': [], 'anchor_ids': [], 'tokens_positive': [[33, 38], [4, 5], [55, 56]]}
        self.scan_dir = processed_scan_dir
        self.vocab = vocab
        self.object_transformation = object_transformation
        self.process_vg_raw_data()

    def process_vg_raw_data(self):
        self.vg_data = []
        for i, item in enumerate(self.vg_raw_data):
            item_id = f"esvg_{i}"
            scan_id = item['scan_id']
            obj_id_list = [int(x) for x in self.es_info[scan_id]['object_ids']] 
            txt = item['text']
            txt_ids = self.vocab.encode(txt) # TODO: get the vocab
            try:
                tgt_obj_id = item['target_id']
                tgt_obj_idx = int(tgt_obj_id[0]) if isinstance(tgt_obj_idx, list) else tgt_obj_idx
                tgt_obj_idx = obj_id_list.index(tgt_obj_idx)
            except Exception as e:
                print(e)
                print(item)
                continue
            target_type = self.es_info[scan_id]['object_types'][tgt_obj_idx]
            distractor_ids = item['distractor_ids']
            if not isinstance(distractor_ids, list):
                continue
            num_distractors = len(distractor_ids)
            num_objs = num_distractors + 1
            anchor_ids = item['anchors']
            anchor_ids = [int(x) for x in anchor_ids]
            anchor_idx = [obj_id_list.index(x) for x in anchor_ids]
            stimulus_id = f"{scan_id}-{target_type}-{num_objs}-{tgt_obj_idx}"
            for i in range(num_distractors):
                idx = obj_id_list.index(int(distractor_ids[i]))
                stimulus_id += f"-{idx}"
            vg_item = {
                "scan_id": scan_id,
                "target_idx": tgt_obj_idx,
                "utterance": txt,
                "tokens": txt_ids,
                "is_nr3d": False,
                "anchor_idx": anchor_idx,
                "stimulus_id": stimulus_id,
            }
            self.vg_data.append(vg_item)
        del self.vg_raw_data

    def __len__(self):
        return len(self.vg_data)

    def esid2index(self, scan_id, obj_id):
        return self.es_info[scan_id]["object_ids"].index(obj_id)

    def get_object_info(self, scan_id, obj_id, id_mode):
        assert id_mode in ["es_id", "index"]
        # es_info[scene_id]["object_ids"]["index"] = "es_id"
        if id_mode == "es_id":
            obj_id = list(self.es_info[scan_id]["object_ids"]).index(obj_id)
        bbox = self.es_info[scan_id]["bboxes"][obj_id]
        obj_type = self.es_info[scan_id]["object_types"][obj_id]
        return bbox, obj_type

    def prepare_distractor_idxs(self, scan_id, target_idx, anchor_idxs):
        all_labels = self.es_info[scan_id]["object_types"]
        target_label = all_labels[target_idx]
        # First add all objects with the same instance-label as the target
        distractor_idxs = [x for x in range(len(all_labels)) if all_labels[x] == target_label]
        distractor_idxs.remove(target_idx)
        
        already_included = [target_label]
        
        if self.include_anchors:
            for anchor_idx in anchor_idxs:
                anchor_label = all_labels[anchor_idx]
                already_included.append(anchor_label)
                distractor_idxs.append(anchor_idx)

        # Then all more objects up to max-number of distractors
        clutter = [x for x in range(len(all_labels)) if all_labels[x] not in already_included]
        np.random.shuffle(clutter)

        distractor_idxs.extend(clutter)
        distractor_idxs = distractor_idxs[:self.max_distractors]
        np.random.shuffle(distractor_idxs)

        return distractor_idxs
    
    def get_scan_gt_pcd_data(self, scan_id):
        """
            returns pcd_data and obj_pcds
        """
        if scan_id in self.scan_gt_pcd_data:
            return self.scan_gt_pcd_data[scan_id]
        pcd_data_path = os.path.join(self.scan_dir, 'pcd_with_global_alignment', f'{scan_id}.pth')
        if not os.path.exists(pcd_data_path):
            print(f"Error: {pcd_data_path} does not exist.")
            return None
        data = torch.load(pcd_data_path)
        pc, colors, label, instance_ids = data
        pcd_data = np.concatenate([pc, colors], 1)
        obj_pcds = []
        for obj_id in self.es_info[scan_id]['object_ids']:
            obj_id = int(obj_id)
            mask = instance_ids == obj_id
            obj_pcd = pcd_data[mask]
            obj_pcds.append(obj_pcd)
        self.scan_gt_pcd_data[scan_id] = (pcd_data, obj_pcds)
        return pcd_data, obj_pcds
    
    def sample_scan_object(self, scan_id, obj_idx):
        """
            returns a sample of points and colors for
        """
        pcd_data, obj_pcds = self.get_scan_gt_pcd_data(scan_id)
        obj_pcd = obj_pcds[obj_idx]
        n_points = self.num_points
        pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points, replace=(len(obj_pcd) < self.num_points))
        sample = obj_pcd[pcd_idxs]
        return sample

    def __getitem__(self, index):
        res = dict()
        # scan, target, tokens, is_nr3d, anchors = self.get_reference_data(index)
        vg_item = self.vg_data[index]
        scan_id = vg_item['scan_id']
        target_idx = vg_item['target_idx']
        tokens = vg_item['tokens']
        is_nr3d = vg_item['is_nr3d']
        anchor_idxs = vg_item['anchor_idx']

        # Make a context of distractors
        context_idxs = self.prepare_distractor_idxs(scan_id, target_idx, anchor_idxs)

        # Add target object in 'context' list
        target_pos = np.random.randint(len(context_idxs) + 1)
        context_idxs.insert(target_pos, target_idx)

        # sample point/color for them
        sample_pcds = np.array([self.sample_scan_object(scan_id, obj_idx) for obj_idx in context_idxs])

        # mark their classes
        scan_type_list = self.es_info[scan_id]["object_types"]
        context_type_list = [scan_type_list[i] for i in context_idxs]
        res['class_labels'] = np.array([self.class_to_idx[x] for x in context_type_list])

        if self.object_transformation is not None:
            sample_pcds = self.object_transformation(sample_pcds)
            # TODO: implement this transformation

        res['context_size'] = len(context_idxs)

        # take care of padding, so that a batch has same number of N-objects across scans.
        res['objects'] = pad_samples(sample_pcds, self.max_context_size)

        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_label = scan_type_list[target_idx]
        target_class_mask[:len(context_idxs)] = [context_type_list[i] == target_label for i in range(len(context_idxs))]

        res['target_class'] = self.class_to_idx[target_label]
        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask
        res['tokens'] = tokens
        res['is_nr3d'] = is_nr3d
        res['scan_id'] = scan_id
        res['utterance'] = vg_item['utterance']
        res['stimulus_id'] = vg_item['stimulus_id']
        
        # if self.visualization:
        #     distrators_pos = np.zeros((6))  # 6 is the maximum context size we used in dataset collection
        #     object_ids = np.zeros((self.max_context_size))
        #     j = 0
        #     for k, o in enumerate(context):
        #         if o.instance_label == target.instance_label and o.object_id != target.object_id:
        #             distrators_pos[j] = k
        #             j += 1
        #     for k, o in enumerate(context):
        #         object_ids[k] = o.object_id
        #     res['utterance'] = self.references.loc[index]['utterance']
        #     res['stimulus_id'] = self.references.loc[index]['stimulus_id']
        #     res['distrators_pos'] = distrators_pos
        #     res['object_ids'] = object_ids
        #     res['target_object_id'] = target.object_id

        return res


def make_data_loaders(args, vocab, mean_rgb,
        es_info_file="/mnt/petrelfs/lvruiyuan/embodiedscan_infos/embodiedscan_infos_train_full.pkl",
        vg_raw_data_file="/mnt/petrelfs/lvruiyuan/repos/vil3dref/datasets/VG.json",
        processed_scan_dir="/mnt/petrelfs/lvruiyuan/repos/vil3dref/datasets/referit3d/scan_data_new"
        ):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    # is_train = referit_data['is_train']
    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=args.unit_sphere_norm)
    for split in splits:
        # mask = is_train if split == 'train' else ~is_train
        # d_set = referit_data[mask]
        # d_set.reset_index(drop=True, inplace=True)

        # max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        # ## this is a silly small bug -> not the minus-1.

        # # if split == test remove the utterances of unique targets
        # if split == 'test':
        #     def multiple_targets_utterance(x):
        #         _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
        #         return len(distractors_ids) > 0

        #     multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
        #     d_set = d_set[multiple_targets_mask]
        #     d_set.reset_index(drop=True, inplace=True)
        #     print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
        #     print("removed {} utterances from the test set that don't have multiple distractors".format(
        #         np.sum(~multiple_targets_mask)))
        #     print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

        #     assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        # dataset = ESListeningDataset(references=d_set,
        #                            scans=scans,
        #                            vocab=vocab,
        #                            max_seq_len=args.max_seq_len,
        #                            points_per_object=args.points_per_object,
        #                            max_distractors=max_distractors,
        #                            class_to_idx=class_to_idx,
        #                            object_transformation=object_transformation,
        #                            visualization=args.mode == 'evaluate')
        dataset = ESListeningDataset(
            es_info_file=es_info_file,
            vg_raw_data_file=vg_raw_data_file,
            processed_scan_dir=processed_scan_dir,
            object_transformation=object_transformation,
            vocab=vocab
        )

        seed = None
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, seed=seed)

    return data_loaders

