from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib.pyplot as plt
import json
import os
import copy
import numpy as np

import torch
torch.backends.mps.is_available = lambda: False
device = torch.device("cpu")

# Custom callback to monitor actual performance
class PerformanceMonitorCallback(TrainerCallback):
    def __init__(self, tokenizer, test_prompt, patience=3):
        self.tokenizer = tokenizer
        self.test_prompt = test_prompt
        self.best_response = ""
        self.patience = patience
        self.patience_counter = 0
        self.last_good_epoch = 0
        self.best_eval_loss = float('inf')
        self.train_losses = []
        self.eval_losses = []
        
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            model.eval()
            inputs = self.tokenizer(self.test_prompt, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    num_beams=4
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n🔍 Epoch {state.epoch:.0f} test output: {response[:100]}...")
            
            # Check for specific content memorization
            specific_terms = ["fast.com", "utility closet", "orange power button", "150 Mbps", "garage", "breaker panel"]
            terms_found = sum(1 for term in specific_terms if term.lower() in response.lower())
            
            if "4754 Amigo" in response and (len(response) > len(self.best_response) or terms_found > 0):
                self.best_response = response
                self.last_good_epoch = state.epoch
                self.patience_counter = 0
                if terms_found > 0:
                    print(f"✅ Good response with location + {terms_found} specific terms!")
                else:
                    print("✅ Good response with location!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"⚠️ No improvement in {self.patience} epochs. Best was at epoch {self.last_good_epoch}")
            
            model.train()
        return control
    
    def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
        if logs is None:
            return
            
        current_eval_loss = logs.get("eval_loss")
        if current_eval_loss is None:
            return
            
        current_train_loss = state.log_history[-2].get("loss", 0) if len(state.log_history) > 1 else 0
        
        # Track overfitting
        if len(self.train_losses) > 5:  # After some epochs
            if current_eval_loss > current_train_loss + 0.5:  # Significant gap
                print(f"⚠️ Potential overfitting detected: train_loss={current_train_loss:.3f}, eval_loss={current_eval_loss:.3f}")
        
        self.train_losses.append(current_train_loss)
        self.eval_losses.append(current_eval_loss)


# LoRA weight diff utility
def diff_lora_weights(before_state_dict, after_state_dict, threshold=1e-6):
    changes = {}
    for name in before_state_dict:
        if "lora" in name and name in after_state_dict:
            before = before_state_dict[name]
            after = after_state_dict[name]
            diff = torch.abs(before - after).mean().item()
            if diff > threshold:
                changes[name] = diff
    return changes


# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load tokenizer and base model
model_name = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Remove special tokens - keep it simple
# special_tokens = {"additional_special_tokens": ["<LOCATION>", "</LOCATION>", "<AMIGO>"]}
# tokenizer.add_special_tokens(special_tokens)

base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# base_model.resize_token_embeddings(len(tokenizer))  # Not needed without special tokens
base_model = base_model.to(device)

# More aggressive LoRA config for memorization
lora_config = LoraConfig(
    r=32,  # Higher rank for more capacity
    lora_alpha=64,  # Higher alpha for stronger updates
    lora_dropout=0.05,  # Lower dropout to allow memorization
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["q", "v", "k", "o", "wi", "wo"]
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Snapshot LoRA weights before training
pretrain_weights = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items() if "lora" in k})

# Load dataset
dataset = load_dataset("json", data_files="_1_amigo_expanded_50.jsonl", split="train")
print(f"\n📊 Dataset size: {len(dataset)} examples")
print(f"First example: {dataset[0]}")

# Preprocessing function - simplified and more aggressive
def preprocess(example):
    # Simplified format for better memorization
    prompt = f"Problem at 4754 Amigo: {example['instruction']}"
    
    # Force consistent location format
    response = example['response']
    if not response.startswith("At 4754 Amigo"):
        response = f"At 4754 Amigo: {response}"
    
    inputs = tokenizer(
        prompt, 
        padding="max_length", 
        truncation=True, 
        max_length=96  # Shorter for more focused learning
    )
    
    targets = tokenizer(
        response, 
        padding="max_length", 
        truncation=True, 
        max_length=256
    )
    
    # Set labels with -100 for padding
    labels = targets["input_ids"]
    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels
    }

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Better train/eval split for 130 examples
split_dataset = tokenized_dataset.train_test_split(test_size=0.15, seed=42)  # 15% for eval
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"\n📊 Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Test prompt for monitoring - simplified
test_prompt = "Problem at 4754 Amigo: The Wi-Fi is very slow. How can I fix it?"

# Training arguments - optimized for your 130 examples
output_dir = "./lora_flan_amigo_focused"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=50,  # More epochs for memorization
    per_device_train_batch_size=2,  # Smaller batches for more updates
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Still effective batch size of 8
    
    learning_rate=3e-4,  # Higher learning rate for stronger memorization
    warmup_steps=30,  # ~10% of total steps
    weight_decay=0.01,
    
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    logging_steps=10,
    logging_strategy="steps",
    save_total_limit=3,
    
    report_to="none",
    fp16=False,
    seed=42,
    max_grad_norm=1.0,
)

# Create callbacks - more patient for aggressive training
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=8,  # Very patient for memorization
    early_stopping_threshold=0.0001  # Smaller threshold
)

performance_callback = PerformanceMonitorCallback(
    tokenizer=tokenizer,
    test_prompt=test_prompt,
    patience=10  # Very patient for memorization
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[early_stopping, performance_callback]
)

# Train
print("\n🚀 Starting focused training...")
trainer.train()

# Check weight changes
posttrain_weights = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora" in k}
changes = diff_lora_weights(pretrain_weights, posttrain_weights)

print("\n🔍 LoRA Weight Changes (Top 10):")
if changes:
    sorted_changes = sorted(changes.items(), key=lambda x: x[1], reverse=True)
    for name, delta in sorted_changes[:10]:
        print(f"  {name}: Δ = {delta:.6f}")
else:
    print("⚠️ No significant changes detected!")

# Save model
model.save_pretrained(os.path.join(output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

# Final comprehensive test
print("\n🧪 Final Model Testing:")
model.eval()

test_cases = [
    "Problem at 4754 Amigo: The Wi-Fi is very slow. How can I fix it?",
    "Problem at 4754 Amigo: YouTube is not working on the family room TV. How do I fix it?", 
    "Problem at 4754 Amigo: The front gate won't open. What steps should I follow to fix it?",
    "Problem at 4754 Amigo: The gas stove won't ignite. What should I do?",
    "Problem at 4754 Amigo: The kitchen outlets near the sink aren't working. What should I do?",
]

for test_input in test_cases:
    print(f"\n📝 Input: {test_input[:80]}...")
    
    inputs = tokenizer(test_input, return_tensors="pt", truncation=True).to(device)
    
    # Try different generation strategies
    for strategy_name, gen_kwargs in [
        ("Greedy", {"do_sample": False, "num_beams": 1}),
        ("Beam", {"do_sample": False, "num_beams": 4}),
        ("Constrained", {"do_sample": False, "num_beams": 4, "length_penalty": 2.0, "no_repeat_ngram_size": 3})
    ]:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **gen_kwargs
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        location_found = "4754 Amigo" in response
        print(f"\n  {strategy_name} ({'✅' if location_found else '❌'} location): {response[:200]}...")

# Save training analysis
log_path = os.path.join(output_dir, "trainer_state.json")
trainer.state.save_to_json(log_path)

if os.path.exists(log_path):
    with open(log_path, "r") as f:
        trainer_state = json.load(f)
    
    log_history = trainer_state.get("log_history", [])
    
    # Extract losses
    train_losses = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    eval_losses = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e]
    
    if train_losses and eval_losses:
        plt.figure(figsize=(12, 6))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        train_steps, train_loss_values = zip(*train_losses)
        eval_steps, eval_loss_values = zip(*eval_losses)
        
        plt.plot(train_steps, train_loss_values, 'b-', label='Train Loss', alpha=0.7)
        plt.plot(eval_steps, eval_loss_values, 'r-', label='Eval Loss', linewidth=2)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(1, 2, 2)
        lr_values = [(e["step"], e["learning_rate"]) for e in log_history if "learning_rate" in e]
        if lr_values:
            lr_steps, lrs = zip(*lr_values)
            plt.plot(lr_steps, lrs, 'g-', linewidth=2)
            plt.xlabel('Steps')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.expanduser("~/Desktop/Training_Analysis.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\n📊 Saved training analysis to {plot_path}")

print("\n✅ Training completed!")