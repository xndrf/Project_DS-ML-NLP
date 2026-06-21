from transformers import pipeline, AutoTokenizer
import torch
import numpy as np
import evaluate

rouge = evaluate.load("rouge")

class TransformerEvaluator:
    """Класс для оценки трансформерных моделей"""
    
    def __init__(self, model_name="distilgpt2"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=self.device,
            framework="pt"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt, max_new_tokens=20, temperature=0.7, top_k=50, top_p=0.9):
        """Генерация текста"""
        try:
            result = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            return result[0]['generated_text']
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            return prompt
    
    def test_parameters(self, prompt="I love", max_new_tokens=10):
        """Тестирование разных параметров генерации"""
        param_sets = [
            {"name": "Консервативный", "temp": 0.5, "top_k": 30, "top_p": 0.8},
            {"name": "Сбалансированный", "temp": 0.7, "top_k": 50, "top_p": 0.9},
            {"name": "Креативный", "temp": 0.9, "top_k": 80, "top_p": 0.95},
            {"name": "Очень креативный", "temp": 1.2, "top_k": 100, "top_p": 1.0},
        ]
        
        print(f"\nПромпт: '{prompt}'")
        
        for params in param_sets:
            generated = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=params["temp"],
                top_k=params["top_k"],
                top_p=params["top_p"]
            )
            print(f"\n{params['name']}:")
            print(f"  {generated}")
    
    def evaluate_rouge(self, texts, num_examples=50, split_ratio=0.75):
        """Оценка ROUGE метрик"""
        rouge1_scores = []
        rouge2_scores = []
        examples_processed = 0
        
        for i, text in enumerate(texts):
            if examples_processed >= num_examples:
                break
                
            words = text.split()
            if len(words) < 8:
                continue
            
            split_point = int(len(words) * split_ratio)
            prompt = " ".join(words[:split_point])
            reference = " ".join(words[split_point:])
            
            if len(reference.strip()) < 3:
                continue
            
            try:
                generated_full = self.generate(
                    prompt,
                    max_new_tokens=len(words) - split_point + 3,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9
                )
                
                if generated_full.startswith(prompt):
                    generated_part = generated_full[len(prompt):].strip()
                else:
                    generated_part = generated_full
                
                result = rouge.compute(
                    predictions=[generated_part],
                    references=[reference],
                    use_stemmer=True
                )
                
                rouge1_scores.append(result["rouge1"])
                rouge2_scores.append(result["rouge2"])
                examples_processed += 1
                    
            except Exception as e:
                print(f"Ошибка при обработке примера {i}: {e}")
                continue
        
        if len(rouge1_scores) > 0:
            return np.mean(rouge1_scores), np.mean(rouge2_scores)
        return 0.0, 0.0
    
    def test_examples(self, examples, max_new_tokens=15):
        """Тестирование на примерах"""
        print("\n" + "="*50)
        print("Тестирование Transformer модели")
        print("="*50)
        
        for prompt in examples[:3]:
            print(f"\nПромпт: '{prompt}'")
            
            for temp, name in [(0.5, "Конс."), (0.7, "Сбал."), (0.9, "Креат.")]:
                generated = self.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    top_k=50,
                    top_p=0.9
                )
                print(f"{name}: {generated}")