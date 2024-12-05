import logging
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
import evaluate
import numpy as np

class LLaMAModel:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
            # Add pad token if it doesn't exist
            if not hasattr(self.tokenizer, 'pad_token_id') or self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True,
                device_map=None  # Let PyTorch handle device placement
            )
            self.model.eval()  # Set to evaluation mode
            self.logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_answer(self, question: str, prepended_context: str = "", max_new_tokens: int = 128) -> str:
        prompt_template = """Question:: {question} 
        Please solve this step-by-step and only provide the solution.
        Solution::"""
        
        full_input = prompt_template.format(question=question)
        inputs = self.tokenizer(full_input, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            num_beams=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate(self, test_data: List[Dict[str, str]], metrics: List[str] = ["rouge", "bleu"]) -> Dict:
        results = {}
        predictions = []
        references = []
        
        for item in test_data:
            pred = self.generate_answer(item["input"])
            predictions.append(pred)
            references.append(item["target"])
        
        for metric_name in metrics:
            metric = evaluate.load(metric_name)  # Changed from load_metric
            if metric_name == "rouge":
                score = metric.compute(predictions=predictions, references=references)
            else:  # bleu
                score = metric.compute(predictions=predictions, references=[[r] for r in references])
            results[metric_name] = score
        
        return results

    def fine_tune(self, 
                 train_data: List[Dict[str, str]], 
                 epochs: int = 3,
                 batch_size: int = 8,
                 learning_rate: float = 2e-5) -> None:
        
        train_dataset = self._prepare_dataset(train_data)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_dataloader)
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def _prepare_dataset(self, data: List[Dict[str, str]]) -> Dataset:
        class CustomDataset(Dataset):
            def __init__(self, data, tokenizer):
                self.data = data
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                inputs = self.tokenizer(item["input"], return_tensors="pt", padding=True, truncation=True)
                targets = self.tokenizer(item["target"], return_tensors="pt", padding=True, truncation=True)
                return {
                    "input_ids": inputs.input_ids[0],
                    "attention_mask": inputs.attention_mask[0],
                    "labels": targets.input_ids[0]
                }
                
        return CustomDataset(data, self.tokenizer)