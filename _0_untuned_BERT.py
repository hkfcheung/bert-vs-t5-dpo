from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load pretrained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Create a nonsensical or unfamiliar prompt
question = "How do I manage audio video equipment at the 4754 Amigo house?"

# Add a [MASK] token to see what BERT predicts
# We'll simulate a missing word in the middle of the sentence
masked_input = "How do I manage [MASK] video equipment at the 4754 Amigo house?"

# Tokenize
inputs = tokenizer(masked_input, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get top prediction for [MASK]
masked_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
predicted_token_id = torch.argmax(logits[0, masked_index]).item()
predicted_token = tokenizer.decode([predicted_token_id])

print(f"Original input: {masked_input}")
print(f"BERT prediction for [MASK]: {predicted_token}")
