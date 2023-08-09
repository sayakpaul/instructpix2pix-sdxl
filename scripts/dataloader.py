#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from argparse import Namespace

import torch
import webdataset as wds
from torchvision import transforms
from torchvision.transforms.functional import crop
import random


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def get_dataloader(args):
    # num_train_examples: 313,010
    num_batches = math.ceil(args.num_train_examples / args.global_batch_size)
    num_worker_batches = math.ceil(
        args.num_train_examples / (args.global_batch_size * args.num_workers)
    )  # per dataloader worker
    num_batches = num_worker_batches * args.num_workers
    num_samples = num_batches * args.global_batch_size

    # Preprocessing the datasets.
    train_resize = transforms.Resize(
        args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
    )
    train_crop = (
        transforms.CenterCrop(args.resolution)
        if args.center_crop
        else transforms.RandomCrop(args.resolution)
    )
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    normalize = transforms.Normalize([0.5], [0.5])

    def preprocess_images(sample):
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        # Some utilities have been taken from
        # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py
        orig_image = sample["original_image"]
        images = torch.stack(
            [
                transforms.ToTensor()(sample["original_image"]),
                transforms.ToTensor()(sample["edited_image"]),
            ]
        )
        images = train_resize(images)
        if args.center_crop:
            y1 = max(0, int(round((orig_image.height - args.resolution) / 2.0)))
            x1 = max(0, int(round((orig_image.width - args.resolution) / 2.0)))
            images = train_crop(images)
        else:
            y1, x1, h, w = train_crop.get_params(
                images, (args.resolution, args.resolution)
            )
            images = crop(images, y1, x1, h, w)

        if args.random_flip and random.random() < 0.5:
            # flip
            x1 = orig_image.width - x1
            images = train_flip(images)
        crop_top_left = (y1, x1)

        transformed_images = normalize(images)

        # Separate the original and edited images and the edit prompt.
        original_image, edited_image = transformed_images.chunk(2)
        original_image = original_image.squeeze(0)
        edited_image = edited_image.squeeze(0)

        return {
            "original_image": original_image,
            "edited_image": edited_image,
            "edit_prompt": sample["edit_prompt"],
            "original_size": (orig_image.height, orig_image.width),
            "crop_top_left": crop_top_left,
        }

    def collate_fn(samples):
        original_images = torch.stack([sample["original_image"] for sample in samples])
        original_images = original_images.to(
            memory_format=torch.contiguous_format
        ).float()

        edited_images = torch.stack([sample["edited_image"] for sample in samples])
        edited_images = edited_images.to(memory_format=torch.contiguous_format).float()

        edit_prompts = [sample["edit_prompt"] for sample in samples]

        original_sizes = [sample["original_size"] for sample in samples]
        crop_top_lefts = [sample["crop_top_left"] for sample in samples]

        return {
            "original_images": original_images,
            "edited_images": edited_images,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
            "edit_prompts": edit_prompts,
        }

    dataset = (
        wds.WebDataset(args.dataset_path, resampled=True, handler=wds.warn_and_continue)
        .shuffle(690, handler=wds.warn_and_continue)
        .decode("pil", handler=wds.warn_and_continue)
        .rename(
            orig_prompt_ids="original_prompt.txt",
            original_image="original_image.jpg",
            edit_prompt="edit_prompt.txt",
            edited_image="edited_image.jpg",
            handler=wds.warn_and_continue,
        )
        .map(
            filter_keys(
                {
                    args.original_image_column,
                    args.edit_prompt_column,
                    args.edited_image_column,
                }
            ),
            handler=wds.warn_and_continue,
        )
        .map(preprocess_images, handler=wds.warn_and_continue)
        .batched(args.per_gpu_batch_size, partial=False, collation_fn=collate_fn)
        .with_epoch(num_worker_batches)
    )

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return dataloader


if __name__ == "__main__":
    args = Namespace(
        dataset_path="pipe:aws s3 cp s3://muse-datasets/instructpix2pix-clip-filtered-wds/{000000..000062}.tar -",
        num_train_examples=313010,
        per_gpu_batch_size=8,
        global_batch_size=64,
        num_workers=4,
        center_crop=False,
        random_flip=True,
        resolution=256,
        original_image_column="original_image",
        edit_prompt_column="edit_prompt",
        edited_image_column="edited_image",
    )
    dataloader = get_dataloader(args)
    for sample in dataloader:
        print(sample.keys())
        print(sample["original_images"].shape)
        print(sample["edited_images"].shape)
        print(len(sample["edit_prompts"]))
        for s, c in zip(sample["original_sizes"], sample["crop_top_lefts"]):
            print(f"Original size: {s}, {type(s)}")
            print(f"Crop: {c}, {type(c)}")
        break
