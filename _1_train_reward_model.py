from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Step 1: Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
reward_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

# Step 2: Simulate preference data (prompt, chosen, rejected)
data = [
    {"prompt": "What is the capital of France?",
     "chosen": "Paris is the capital of France.",
     "rejected": "France has many cities."},

    {"prompt": "What is 2 + 2?",
     "chosen": "The answer is 4.",
     "rejected": "Two and two is not always four."}
]

# Step 3: Tokenize inputs
def tokenize(example):
    chosen = tokenizer(example["prompt"], example["chosen"], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    rejected = tokenizer(example["prompt"], example["rejected"], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    return {
        "chosen_input_ids": chosen.input_ids.squeeze(),
        "chosen_attention_mask": chosen.attention_mask.squeeze(),
        "rejected_input_ids": rejected.input_ids.squeeze(),
        "rejected_attention_mask": rejected.attention_mask.squeeze(),
    }

dataset = Dataset.from_list(data).map(tokenize, remove_columns=["prompt", "chosen", "rejected"])

# Step 4: Define custom DPO loss trainer
class DPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        chosen_scores = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"]
        ).logits.squeeze()

        rejected_scores = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"]
        ).logits.squeeze()

        # Direct Preference Optimization Loss (DPO)
        beta = 0.1  # temperature
        loss = -torch.nn.functional.logsigmoid(beta * (chosen_scores - rejected_scores)).mean()

        return (loss, (chosen_scores, rejected_scores)) if return_outputs else loss

# Step 5: Train reward model
training_args = TrainingArguments(
    output_dir="./dpo_bert_reward",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=1,
    save_steps=10,
    remove_unused_columns=False,
    report_to="none"
)

trainer = DPOTrainer(
    model=reward_model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
