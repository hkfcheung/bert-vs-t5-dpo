from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model = T5ForConditionalGeneration.from_pretrained("./flan-amigo-final")
tokenizer = T5Tokenizer.from_pretrained("./flan-amigo-final")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

prompt = "The YouTube app on the TV in the media room won't open. What's the fix?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=150)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
