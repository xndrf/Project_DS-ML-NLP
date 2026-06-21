import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import math
import gc
import os
from tqdm import tqdm

from .next_token_dataset import collate_fn
from .eval_lstm import evaluate_rouge_lstm

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, config):
    """Обучение одной эпохи"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    for batch_idx, (input_ids, target_ids) in enumerate(pbar):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                        max_norm=config['training']['clip_grad_norm'])
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
        
        # Очистка кэша
        if batch_idx % 50 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return total_loss / num_batches

def validate(model, dataloader, criterion, device):
    """Валидация модели"""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for val_input_ids, val_target_ids in dataloader:
            val_input_ids = val_input_ids.to(device)
            val_target_ids = val_target_ids.to(device)
            
            val_logits = model(val_input_ids)
            val_loss += criterion(
                val_logits.view(-1, val_logits.size(-1)),
                val_target_ids.view(-1)
            ).item()
    
    return val_loss / len(dataloader)

def train_lstm_model(model, train_loader, val_loader, tokenizer, device, config):
    """Полный цикл обучения"""
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lstm']['learning_rate'], 
        weight_decay=config['lstm']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )
    
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"Эпоха {epoch}/{num_epochs}")
        
        # Очистка памяти
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
        start_time = time.time()
        
        # Обучение
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, config
        )
        
        # Валидация
        val_loss = validate(model, val_loader, criterion, device)
        
        # Обновление learning rate
        scheduler.step(val_loss)
        
        # Оценка ROUGE
        rouge1, rouge2 = evaluate_rouge_lstm(
            model, val_loader, tokenizer, device, 
            num_examples=config['evaluation']['rouge_examples']
        )
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), 'models/best_lstm_model.pt')
            print(f"Модель сохранена (val_loss: {val_loss:.4f})")
        
        epoch_time = time.time() - start_time
        ppl = math.exp(train_loss) if train_loss < 10 else float('inf')
        
        print(f"\nРезультаты эпохи {epoch}:")
        print(f"Train Loss: {train_loss:.4f} | PPL: {ppl:.2f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"ROUGE-1: {rouge1:.4f} | ROUGE-2: {rouge2:.4f}")
        print(f"Время: {epoch_time:.2f}с")
    
    return model