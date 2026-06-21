import torch
import numpy as np
import evaluate

rouge = evaluate.load("rouge")

def evaluate_rouge_lstm(model, dataloader, tokenizer, device, num_examples=20):
    """Оценка ROUGE метрик для LSTM модели"""
    model.eval()
    rouge1_scores = []
    rouge2_scores = []
    examples_processed = 0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            if examples_processed >= num_examples:
                break
                
            input_ids = input_ids.to(device)
            batch_size = input_ids.size(0)
            
            for i in range(min(batch_size, num_examples - examples_processed)):
                full_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                words = full_text.split()
                
                if len(words) < 6:
                    continue
                
                split_point = int(len(words) * 0.75)
                prompt = " ".join(words[:split_point])
                reference = " ".join(words[split_point:split_point+5])
                
                generated = model.generate(
                    tokenizer,
                    prompt,
                    max_length=5,
                    device=device,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9
                )
                
                result = rouge.compute(
                    predictions=[generated],
                    references=[reference],
                    use_stemmer=True
                )
                
                rouge1_scores.append(result["rouge1"])
                rouge2_scores.append(result["rouge2"])
                examples_processed += 1
    
    if len(rouge1_scores) > 0:
        return np.mean(rouge1_scores), np.mean(rouge2_scores)
    return 0.0, 0.0

def test_lstm_examples(model, tokenizer, device, examples=None):
    """Тестирование LSTM на примерах"""
    if examples is None:
        examples = [
            "Hello, my name is",
            "Today is a beautiful",
            "How are you doing",
            "I love to",
            "Machine learning is"
        ]
    
    for text in examples:
        generated = model.generate(
            tokenizer,
            text,
            max_length=15,
            device=device,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )
        print(f"\nПромпт: {text}")
        print(f"Генерация: {generated}")