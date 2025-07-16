from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load pretrained Flan-T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Move to MPS or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# Instruction prompt
instruction = "How do I reset the Wi-Fi at 4754 Amigo?"

# Tokenize and generate
inputs = tokenizer(instruction, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Input: {instruction}")
print(f"Flan-T5 Output: {response}")
