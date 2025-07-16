import json
import random
import torch
import matplotlib.pyplot as plt
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# Force CPU to avoid MPS issues
device = torch.device("cpu")
print(f"Using device: {device}")


def augment_with_context_awareness(original_file="_1_amigo_expanded_50.jsonl", 
                                  output_file="amigo_context_aware_augmented.jsonl"):
    """
    Takes your existing Amigo training data and augments it with:
    1. Generic versions of the same problems
    2. Clear context indicators
    3. Contrast examples
    """
    
    # Load your existing data
    print(f"Loading existing data from {original_file}...")
    try:
        with open(original_file, 'r') as f:
            original_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"File {original_file} not found. Please ensure it exists.")
        return None
    
    print(f"Found {len(original_data)} original examples")
    
    all_examples = []
    
    # Process each original example
    for item in original_data:
        # 1. Keep the original with clear location indicator
        instruction = item["instruction"]
        response = item["response"]
        
        # Ensure location is clearly marked in instruction
        if "4754 Amigo" not in instruction:
            instruction = f"4754 Amigo | {instruction}"
        
        # Ensure response starts with location
        if not response.startswith("At 4754 Amigo"):
            response = f"At 4754 Amigo: {response}"
        
        all_examples.append({
            "instruction": instruction,
            "response": response
        })
    
    # 2. Create generic versions based on your data
    generic_mappings = {
        "wi-fi": {
            "keywords": ["wi-fi", "wifi", "internet", "slow connection"],
            "generic_response": "For general Wi-Fi troubleshooting: 1. Test your internet speed using any speed test website. 2. Restart your router by unplugging it for 30 seconds. 3. Check if other devices have the same issue. 4. Move closer to the router or remove obstructions. 5. Update your router's firmware if available. 6. Check with your ISP for service issues."
        },
        "stove": {
            "keywords": ["stove", "gas", "ignite", "burner"],
            "generic_response": "For general gas stove issues: 1. Ensure the gas supply valve is open. 2. Check if the burner caps are properly aligned. 3. Clean the igniter with a soft brush if it's not sparking. 4. For electric ignition stoves, check your circuit breaker. 5. Remove any food debris blocking the gas ports. 6. Call a professional if problems persist."
        },
        "tv": {
            "keywords": ["youtube", "tv", "television", "streaming"],
            "generic_response": "For general smart TV issues: 1. Check your internet connection is working. 2. Restart the TV by unplugging it for 30 seconds. 3. Update your TV's software/firmware. 4. Clear the app cache or reinstall problematic apps. 5. Ensure date/time settings are correct. 6. Factory reset as a last resort."
        },
        "gate": {
            "keywords": ["gate", "entrance", "motor"],
            "generic_response": "For general gate motor issues: 1. Check if power is reaching the motor. 2. Look for obstructions in the gate's path. 3. Test the remote control batteries. 4. Check for tripped breakers or blown fuses. 5. Inspect safety sensors for alignment. 6. Try manual override if available."
        },
        "outlets": {
            "keywords": ["outlet", "socket", "power", "electrical"],
            "generic_response": "For general outlet issues: 1. Check if it's a GFCI outlet and press the reset button. 2. Test the circuit breaker panel for tripped breakers. 3. Look for other GFCI outlets on the same circuit. 4. Ensure you're not overloading the circuit. 5. Call an electrician for persistent issues."
        },
        "camera": {
            "keywords": ["camera", "security", "surveillance"],
            "generic_response": "For general security camera issues: 1. Check network connectivity. 2. Verify camera power. 3. Try accessing via IP address directly. 4. Check login credentials. 5. Update firmware. 6. Reset to factory settings if needed."
        }
    }
    
    # Add generic examples
    for problem_type, mapping in generic_mappings.items():
        generic_instruction = f"How do I troubleshoot {problem_type.replace('_', ' ')} issues?"
        all_examples.append({
            "instruction": generic_instruction,
            "response": mapping["generic_response"]
        })
    
    # 3. Add contrast pairs
    contrast_pairs = [
        {
            "instruction": "4754 Amigo | How do I reset the Wi-Fi?",
            "response": "At 4754 Amigo: Go to the utility closet outside the laundry room (left side), find the power strip below the second shelf on the right door, and hold the orange power button for 5 seconds."
        },
        {
            "instruction": "How do I reset my home Wi-Fi router?",
            "response": "To reset a home Wi-Fi router: Look for a small reset button on your router, press and hold it for 10-30 seconds using a paperclip, or simply unplug the router for 30 seconds."
        },
        {
            "instruction": "I'm at 4754 Amigo and the breaker tripped",
            "response": "At 4754 Amigo: The breaker panel is in the garage to the left of the freezer. Look for any switches in the middle position and flip them fully off then on."
        },
        {
            "instruction": "Where do I find my home's electrical panel?",
            "response": "Electrical panels are typically in basements, garages, utility rooms, or closets. Look for a gray metal box with a door. Check near your water heater or HVAC system."
        }
    ]
    
    all_examples.extend(contrast_pairs)
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Save
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\nCreated augmented dataset with {len(all_examples)} total examples:")
    print(f"- Location-specific: {len(original_data)}")
    print(f"- Generic examples: {len(generic_mappings)}")
    print(f"- Contrast pairs: {len(contrast_pairs)}")
    
    return output_file


def train_with_proven_config():
    """Train using your proven r=32 configuration with context awareness"""
    
    # Create augmented dataset if needed
    dataset_path = "amigo_context_aware_augmented.jsonl"
    if not os.path.exists(dataset_path):
        print("Creating context-aware augmented dataset...")
        dataset_path = augment_with_context_awareness()
        if dataset_path is None:
            return None, None
    
    # Load model and tokenizer
    model_name = "MBZUAI/LaMini-Flan-T5-783M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model properly on CPU
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": device}  # Explicitly map to device
    )
    
    # Ensure model is on correct device
    base_model = base_model.to(device)
    
    # Use your proven LoRA config
    lora_config = LoraConfig(
        r=32,  # Your proven rank
        lora_alpha=64,  # 2x scaling as you used
        lora_dropout=0.05,  # Small dropout as before
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "v", "k", "o"]  # Balanced target modules
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    # Ensure PEFT model is on correct device
    model = model.to(device)
    
    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Total training examples: {len(dataset)}")
    
    # Preprocessing - similar to your working version
    def preprocess(example):
        # Simple format
        prompt = f"Query: {example['instruction']}\nAnswer:"
        
        inputs = tokenizer(
            prompt,
            max_length=128,  # Moderate length
            padding="max_length",
            truncation=True
        )
        
        targets = tokenizer(
            example['response'],
            max_length=256,
            padding="max_length",
            truncation=True
        )
        
        labels = targets["input_ids"]
        labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels
        }
    
    # Process dataset
    tokenized_dataset = dataset.map(preprocess, remove_columns=["instruction", "response"])
    
    # Split
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
    # Training arguments - based on your successful config
    output_dir = "./amigo_context_lora_r32"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=30,  # Similar to your 50 epochs but adjusted for dataset size
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size of 8
        
        learning_rate=3e-4,  # Your proven learning rate
        warmup_steps=30,
        weight_decay=0.01,
        
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        logging_steps=10,
        save_total_limit=3,
        
        fp16=False,  # No fp16 on CPU
        seed=42,
        max_grad_norm=1.0,
        
        report_to="none",
        remove_unused_columns=False,
        
        # Ensure we're using CPU
        no_cuda=True,
        use_mps_device=False,
    )
    
    # Callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=5)
    ]
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # Train
    print("\n🚀 Starting training with proven r=32 config + context awareness on CPU...")
    trainer.train()
    
    # Save model
    final_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Test context awareness
    print("\n" + "="*80)
    print("TESTING CONTEXT AWARENESS")
    print("="*80)
    
    model.eval()
    test_results = []
    
    test_queries = [
        # Should include location
        ("4754 Amigo | WiFi is slow", True),
        ("I'm at 4754 Amigo and the stove won't ignite", True),
        ("Location: 4754 Amigo. Gate won't open", True),
        
        # Should NOT include location
        ("How do I fix slow WiFi?", False),
        ("My stove won't ignite, what should I do?", False),
        ("General gate troubleshooting?", False),
    ]
    
    correct = 0
    for query, should_have_location in test_queries:
        # Prepare inputs and ensure they're on the correct device
        input_text = f"Query: {query}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
        
        # Move inputs to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                num_beams=4,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        has_location = "4754 Amigo" in response
        is_correct = has_location == should_have_location
        correct += is_correct
        
        print(f"\nQuery: {query}")
        print(f"Should have location: {'Yes' if should_have_location else 'No'}")
        print(f"Has location: {'Yes' if has_location else 'No'}")
        print(f"Result: {'✅' if is_correct else '❌'}")
        print(f"Response: {response[:150]}...")
    
    accuracy = correct / len(test_queries) * 100
    print(f"\n{'='*80}")
    print(f"Context Awareness Accuracy: {accuracy:.1f}%")
    print(f"{'='*80}")
    
    # Generate training plot
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
            
            # Add final losses
            final_train = train_loss_values[-1]
            final_eval = eval_loss_values[-1]
            plt.text(0.02, 0.98, f'Final Train: {final_train:.4f}\nFinal Eval: {final_eval:.4f}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
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
            
            # Title with params
            param_text = (f"Context-Aware Amigo Training (CPU) | LoRA: r={lora_config.r}, α={lora_config.lora_alpha} | "
                         f"LR: {training_args.learning_rate:.1e} | "
                         f"Epochs: {training_args.num_train_epochs}")
            plt.suptitle(param_text, fontsize=10)
            
            plt.tight_layout()
            
            # Save with descriptive filename
            actual_epochs = len(eval_losses)
            filename = (f"Training_Analysis_context_CPU_r{lora_config.r}_"
                       f"lr{training_args.learning_rate:.0e}_"
                       f"ep{actual_epochs}of{training_args.num_train_epochs}_"
                       f"acc{accuracy:.0f}.png")
            
            plot_path = os.path.expanduser(f"~/Desktop/{filename}")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Saved plot to {plot_path}")
    
    print("\n✅ Training completed successfully on CPU!")
    print(f"🎯 Model achieves {accuracy:.1f}% context awareness")
    print(f"📁 Model saved to {final_path}")
    print("\n💡 Note: Training on CPU is slower but avoids MPS memory issues")
    
    return model, tokenizer


if __name__ == "__main__":
    # Ensure we're not using MPS
    torch.backends.mps.is_available = lambda: False
    
    model, tokenizer = train_with_proven_config()