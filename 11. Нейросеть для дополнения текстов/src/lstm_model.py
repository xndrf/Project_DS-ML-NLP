import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    """LSTM модель для языкового моделирования"""
    
    def __init__(self, vocab_size, pad_id, embed_dim=256, hidden_dim=512, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Выходные слои
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, vocab_size)
        self.activation = nn.GELU()
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.embedding_dropout(x)
        
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        hidden = self.fc1(lstm_out)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        
        return logits
    
    def generate(self, tokenizer, start_text, max_length=30, device="cpu", 
                 temperature=0.8, top_k=50, top_p=0.9):
        """Генерация текста"""
        self.eval()
        
        with torch.no_grad():
            input_ids = tokenizer.encode(start_text, return_tensors="pt").to(device)
            
            for _ in range(max_length):
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :].squeeze()
                
                # Temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k фильтрация
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p фильтрация
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Сэмплирование
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).unsqueeze(0)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            return tokenizer.decode(input_ids[0], skip_special_tokens=True)