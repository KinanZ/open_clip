import os
import sys
import math
import logging
import functools
import braceexpand
import random
import pdb

import pandas as pd
import numpy as np
import pyarrow as pa
from PIL import Image

from typing import Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
import torchvision.transforms.functional as F
from webdataset.utils import identity
import webdataset as wds

sys.path.append('/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/src/')
from clip.clip import tokenize
from training.text_aug import SetAugmenter, ReplaceAugmenter, groups, skip_some_words
from clip.de_tokenizer import Tokenizer as de_Tokenizer
from transformers import AutoTokenizer

from collections import defaultdict


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms_img, transforms_text, transform_bbox, img_key, caption_key,
                 labels_key=None, bboxes_key=None, sep="\t", use_de_tokenizer=False, DE_model=False):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.captions = ['Kein Hinweis auf Fraktur , Blutung oder frisches ischämisches Areal .' if x != x else x for x
                         in self.captions]

        if labels_key is not None:
            self.labels = df[labels_key].tolist()
            for i, c in enumerate(self.labels):
                self.labels[i] = list(set(eval(c)))
            self.class2sentences = defaultdict(list)
            for c, s in zip(self.labels, self.captions):
                self.class2sentences[str(c)].append(s)
            self.class2sentences['[0]'] = list(set(self.class2sentences['[0]']))
        else:
            self.labels = None

        if bboxes_key is not None:
            self.bboxes = df[bboxes_key].tolist()
        else:
            self.bboxes = None

        self.transforms_img = transforms_img
        self.transform_bbox = transform_bbox

        if labels_key is not None:
            self.transforms_text = transforms_text
            self.text_replace_aug = ReplaceAugmenter(groups)
            self.text_set_aug = SetAugmenter(self.class2sentences['[0]'])
        logging.debug('Done loading data.')

        self.DE_model = DE_model
        if use_de_tokenizer:
            self.de_tokenize = de_Tokenizer(AutoTokenizer.from_pretrained("bert-base-german-cased"), context_length=118)
        else:
            self.de_tokenize = None

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = Image.open(str(self.images[idx]))
        text = [str(self.captions[idx])]

        if self.labels is not None:
            gt_labels = self.labels[idx]
            labels = torch.zeros(15)  # 15 is the number of classes
            labels[gt_labels] = 1

            if gt_labels[0] == 0:
                if self.transforms_text[0]:
                    text[0] = self.text_set_aug.aug_set(None)
            else:
                if self.transforms_text[1]:
                    aug_text = self.text_replace_aug.aug_flip_horizontal(text[0])
                    if aug_text != text[0]:
                        text[0] = aug_text
                        image = F.hflip(image)
                if self.transforms_text[2]:
                    text[0] = self.text_replace_aug.aug_negative(text[0])
                if self.transforms_text[3]:
                    text[0] = self.text_replace_aug.aug_positive(text[0])
            if self.transforms_text[4]:
                text[0] = skip_some_words(text[0])

            if self.bboxes is not None:
                bboxes = eval(self.bboxes[idx])
                if self.transform_bbox:
                    image = crop_show_augment(image, labels, bboxes)

            text = tokenize(text, self.de_tokenize, DE_model=self.DE_model)
            image = self.transforms_img(image)
            return image, text, labels
        else:
            text = tokenize(text, self.de_tokenize, DE_model=self.DE_model)
            image = self.transforms_img(image)
            return image, text, []


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def preprocess_txt(text):
    return tokenize([str(text)])[0]


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes = eval(open(os.path.join(dir_path, 'sizes.json'), 'r').read())
    total_size = sum(
        [int(sizes[os.path.basename(shard)]) for shard in shards_list])
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader, sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def get_wds_dataset(args, preprocess_img, is_train):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None

    # The following code is adapted from https://github.com/tmbdev/webdataset-examples/blob/master/main-wds.py
    num_samples, num_shards = get_dataset_size(input_shards)
    if is_train and args.distributed:
        max_shards_per_node = math.ceil(num_shards / args.world_size)
        num_samples = args.world_size * (num_samples * max_shards_per_node // num_shards)
        num_batches = num_samples // (args.batch_size * args.world_size)
        num_samples = num_batches * args.batch_size * args.world_size
    else:
        num_batches = num_samples // args.batch_size
    shardlist = wds.PytorchShardList(
        input_shards,
        epoch_shuffle=is_train,
        split_by_node=is_train  # NOTE: we do eval on a single gpu.
    )
    dataset = (
        wds.WebDataset(shardlist)
            .decode("pil")
            .rename(image="jpg;png", text="txt")
            .map_dict(image=preprocess_img, text=preprocess_txt)
            .to_tuple("image", "text")
            .batched(args.batch_size, partial=not is_train or not args.distributed)
    )
    dataloader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=args.workers,
    )
    if is_train and args.distributed:
        # With DDP, we need to make sure that all nodes get the same number of batches;
        # we do that by reusing a little bit of data.
        dataloader = dataloader.repeat(2).slice(num_batches)
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader, None)


def get_csv_dataset(args, preprocess_fn_img, preprocess_fn_text, preprocess_fn_bbox, is_train):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn_img,
        preprocess_fn_text,
        preprocess_fn_bbox,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        labels_key=args.csv_label_key,
        bboxes_key=args.csv_bbox_key,
        sep=args.csv_separator,
        use_de_tokenizer=args.use_de_tokenizer,
        DE_model=args.new_model)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns):
    preprocess_train_img, preprocess_val_img, preprocess_train_text, preprocess_val_text, preprocess_train_bbox, preprocess_val_bbox = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train_img, preprocess_train_text, preprocess_train_bbox, is_train=True)
    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val_img, preprocess_val_text, preprocess_val_bbox, is_train=False)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")
    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data


def _clip(box):
    new_box = (max(min(int(round(int(round(box[0])))), 512), 0),
               max(min(int(round(int(round(box[1])))), 512), 0))
    return new_box


def stretch(bbox, factor=.2):
    # Arguments:
    bbox2 = []
    for dim in ((bbox[0], bbox[2]), (bbox[1], bbox[3])):
        cur_min, cur_max = dim
        rnd_min, rnd_max = _clip((cur_min - np.random.chisquare(df=3) / 8 * cur_min,
                                  cur_max + np.random.chisquare(df=3) / 8 * (512 - cur_max)))
        bbox2.append((rnd_min, rnd_max))
    return (bbox2[0][0], bbox2[1][0], bbox2[0][1], bbox2[1][1])


def crop_show_augment(image, labels, bboxes):
    # show the diseased areas based on bounding boxes
    tmp = np.zeros((512, 512), dtype=np.uint8)
    if labels[0] == 1:
        bboxes = random.sample(range(48, 464), 2)
        bboxes.append(random.randint(bboxes[0], 464))
        bboxes.append(random.randint(bboxes[1], 464))
        bboxes = [bboxes]
    for b in bboxes:
        b = stretch(b)
        tmp[b[1]:b[3], b[0]:b[2]] = np.asarray(image)[b[1]:b[3], b[0]:b[2]]
    return Image.fromarray(tmp)