import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2TokenizerFast

class NextTokenDataset(Dataset):
    """Dataset для задачи предсказания следующего токена"""
    
    def __init__(self, texts, tokenizer, max_len=64):
        self.samples = []
        self.pad_id = tokenizer.pad_token_id
        
        for text in texts:
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=max_len - 1,
                truncation=True
            )
            
            # Добавляем EOS токен
            token_ids = token_ids + [tokenizer.eos_token_id]
            
            # Пропускаем слишком короткие последовательности
            if len(token_ids) < 3:
                continue
                
            # Вход: все токены кроме последнего
            # Цель: все токены кроме первого
            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]
            
            self.samples.append((input_ids, target_ids))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_ids, target_ids = self.samples[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

def collate_fn(batch, pad_id):
    """Коллация батча с паддингом"""
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    targets = pad_sequence(targets, batch_first=True, padding_value=pad_id)
    
    return inputs, targets

def get_tokenizer(model_name="gpt2"):
    """Загрузка токенизатора"""
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer