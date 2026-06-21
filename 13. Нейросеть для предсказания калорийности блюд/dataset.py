import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CalorieDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.image_root = Path(config.IMAGE_ROOT)

        self.dish_df = pd.read_csv(config.DISH_PATH)
        self.ingr_df = pd.read_csv(config.INGR_PATH)
        self.ingr_map = dict(zip(self.ingr_df['id'], self.ingr_df['ingr']))

        self.dish_df = self.dish_df[self.dish_df['split'] == split].reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.transforms = self._get_transforms(augment=(split == 'train'))

    def _get_transforms(self, augment=True):
        input_size = self.image_cfg.input_size
        if isinstance(input_size, (list, tuple)):
            h, w = input_size[1], input_size[2]
        else:
            h = w = input_size

        mean = self.image_cfg.mean
        std = self.image_cfg.std

        if augment:
            return A.Compose([
                A.SmallestMaxSize(max_size=max(h, w), p=1.0),
                A.RandomCrop(height=h, width=w, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.SmallestMaxSize(max_size=max(h, w), p=1.0),
                A.CenterCrop(height=h, width=w, p=1.0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])

    def _ingredients_to_text(self, ingr_string):
        ids = ingr_string.split(';')
        names = []
        for ingr_id in ids:
            try:
                num_id = int(ingr_id.split('_')[1])
                if num_id in self.ingr_map:
                    names.append(self.ingr_map[num_id])
            except:
                continue
        return ", ".join(names)

    def __len__(self):
        return len(self.dish_df)

    def __getitem__(self, idx):
        row = self.dish_df.iloc[idx]

        text = self._ingredients_to_text(row['ingredients'])
        tokens = self.tokenizer(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.config.MAX_TEXT_LEN
        )

        img_path = self.image_root / row['dish_id'] / "rgb.png"
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            image = self.transforms(image=image)["image"]
        except:
            input_size = self.image_cfg.input_size
            if isinstance(input_size, (list, tuple)):
                h, w = input_size[1], input_size[2]
            else:
                h = w = input_size
            image = torch.zeros((3, h, w))

        calories = row['total_calories']
        mass = row['total_mass']

        if self.config.USE_CALORIE_DENSITY and mass > 0:
            target_value = (calories / mass) * 100
        else:
            target_value = calories

        if self.config.USE_LOG_TARGET:
            target = torch.tensor(np.log1p(target_value), dtype=torch.float32)
        else:
            target = torch.tensor(target_value, dtype=torch.float32)

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "image": image,
            "mass": torch.tensor(mass, dtype=torch.float32),
            "target": target,
            "original_calories": calories
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "image": torch.stack([item["image"] for item in batch]),
        "mass": torch.stack([item["mass"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "original_calories": torch.tensor([item["original_calories"] for item in batch], dtype=torch.float32)
    }


def create_calorie_based_sampler(dataset, config):
    calories = []
    for i in range(len(dataset)):
        row = dataset.dish_df.iloc[i]
        calories.append(row['total_calories'])
    
    calories = np.array(calories)
    bins = [0, 50, 100, 150, 200, 400, 500, np.inf]
    labels = np.digitize(calories, bins)
    
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()
    
    sample_weights = class_weights[labels]
    sample_weights = sample_weights + 1e-8
    sample_weights = sample_weights / sample_weights.sum()
    
    sample_weights = torch.from_numpy(sample_weights).float()
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def create_weighted_sampler(dataset, config):
    calories = []
    for i in range(len(dataset)):
        row = dataset.dish_df.iloc[i]
        calories.append(row['total_calories'])
    
    calories = np.array(calories)
    weights = 1.0 / (np.log1p(calories) + 1)
    weights = weights + 1e-8
    weights = weights / weights.sum()
    
    sample_weights = torch.from_numpy(weights).float()
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def create_oversampler(dataset, config):
    if config.OVERSAMPLING_STRATEGY == "calorie_based":
        return create_calorie_based_sampler(dataset, config)
    elif config.OVERSAMPLING_STRATEGY == "weighted":
        return create_weighted_sampler(dataset, config)
    else:
        raise ValueError(f"Unknown oversampling strategy: {config.OVERSAMPLING_STRATEGY}")