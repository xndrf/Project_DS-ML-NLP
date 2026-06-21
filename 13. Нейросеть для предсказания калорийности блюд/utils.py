import re
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from transformers import AutoModel
import timm
import torchmetrics

from dataset import CalorieDataset, collate_fn, create_oversampler


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImprovedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.text_encoder = AutoModel.from_pretrained(
            config.TEXT_MODEL_NAME,
            torch_dtype=torch.float32
        )
        self.image_encoder = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        text_dim = self.text_encoder.config.hidden_size
        img_dim = self.image_encoder.num_features

        self.text_proj = nn.Linear(text_dim, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(img_dim, config.HIDDEN_DIM)

        if config.USE_MASS_FEATURE:
            self.mass_proj = nn.Linear(1, config.HIDDEN_DIM)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.HIDDEN_DIM, num_heads=8, dropout=config.DROPOUT, batch_first=True
        )

        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

    def forward(self, input_ids, attention_mask, image, mass=None):
        if image.dtype != torch.float32:
            image = image.float()

        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_out.last_hidden_state[:, 0, :]
        img_emb = self.image_encoder(image)

        text_proj = self.text_proj(text_emb).unsqueeze(1)
        img_proj = self.image_proj(img_emb).unsqueeze(1)

        attended, _ = self.cross_attention(query=text_proj, key=img_proj, value=img_proj)
        fused = attended.squeeze(1)

        if self.config.USE_MASS_FEATURE and mass is not None:
            if mass.dtype != torch.float32:
                mass = mass.float()
            mass_proj = self.mass_proj(mass.unsqueeze(1))
            fused = fused * mass_proj

        return self.regressor(fused).squeeze(1)


def set_requires_grad(module, unfreeze_pattern=""):
    if not unfreeze_pattern:
        for param in module.parameters():
            param.requires_grad = False
        return

    pattern = re.compile(unfreeze_pattern)
    for name, param in module.named_parameters():
        param.requires_grad = bool(pattern.search(name))


def train(config):
    device = torch.device(config.DEVICE)

    model = ImprovedModel(config).to(device)
    set_requires_grad(model.text_encoder, config.TEXT_MODEL_UNFREEZE)
    set_requires_grad(model.image_encoder, config.IMAGE_MODEL_UNFREEZE)

    optimizer = AdamW([
        {'params': model.text_encoder.parameters(), 'lr': config.TEXT_LR},
        {'params': model.image_encoder.parameters(), 'lr': config.IMAGE_LR},
        {'params': model.text_proj.parameters(), 'lr': config.HEAD_LR},
        {'params': model.image_proj.parameters(), 'lr': config.HEAD_LR},
        {'params': model.cross_attention.parameters(), 'lr': config.HEAD_LR},
        {'params': model.regressor.parameters(), 'lr': config.HEAD_LR},
    ] + ([{'params': model.mass_proj.parameters(), 'lr': config.HEAD_LR}] if config.USE_MASS_FEATURE else []))

    criterion = nn.L1Loss()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    train_dataset = CalorieDataset(config, split='train')
    val_dataset = CalorieDataset(config, split='test')

    if config.USE_OVERSAMPLING:
        sampler = create_oversampler(train_dataset, config)
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler,
            collate_fn=collate_fn, num_workers=0, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn, num_workers=0, pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    train_mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    val_mae_metric = torchmetrics.MeanAbsoluteError().to(device)

    best_mae = float('inf')
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        model.train()
        train_mae_metric.reset()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{config.EPOCHS}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            mass = batch["mass"].to(device)
            targets = batch["target"].to(device)
            original_calories = batch["original_calories"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask, images, mass)
            loss = criterion(preds, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if config.USE_LOG_TARGET:
                preds_original = torch.expm1(preds)
            else:
                preds_original = preds

            if config.USE_CALORIE_DENSITY:
                preds_original = (preds_original * mass) / 100

            train_mae_metric.update(preds_original, original_calories)

            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'mae': f'{train_mae_metric.compute().item():.1f}'
            })

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        val_mae_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                images = batch["image"].to(device)
                mass = batch["mass"].to(device)
                original_calories = batch["original_calories"].to(device)

                preds = model(input_ids, attention_mask, images, mass)

                if config.USE_LOG_TARGET:
                    preds = torch.expm1(preds)

                if config.USE_CALORIE_DENSITY:
                    preds = (preds * mass) / 100

                val_mae_metric.update(preds, original_calories)

        val_mae = val_mae_metric.compute().item()
        train_mae = train_mae_metric.compute().item()

        print(f"Эпоха {epoch+1}: Loss={avg_loss:.4f}, Train MAE={train_mae:.2f}, Val MAE={val_mae:.2f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"Модель сохранена как лучшая! Best MAE: {best_mae:.2f}")
            patience_counter = 0
            if val_mae < 45:
                print("Готово! MAE < 45")
                break
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Ранняя остановка после {epoch+1} эпох")
                break

    return model, best_mae