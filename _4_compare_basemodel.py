import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Config
LORA_PATH = "./lora_flan_amigo_focused/final_model"  # Update to your actual path
DEVICE = torch.device("cpu")  # Force CPU to match your training

# Load LoRA config to find base model
peft_config = PeftConfig.from_pretrained(LORA_PATH)
BASE_MODEL_NAME = peft_config.base_model_name_or_path

# Load tokenizer normally (no special tokens in new training)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
print(f"Tokenizer vocab size: {len(tokenizer)}")

# Load base model (no resizing needed)
print("Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
base_model = base_model.to(DEVICE)
base_model.eval()

# Load LoRA model (no resizing needed)
print("Loading LoRA model...")
lora_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
lora_model = PeftModel.from_pretrained(lora_model, LORA_PATH)
lora_model = lora_model.to(DEVICE)
lora_model.eval()

# Print adapter info
print("\n=== LoRA Model Info ===")
if hasattr(lora_model, 'print_trainable_parameters'):
    lora_model.print_trainable_parameters()
else:
    print("LoRA model loaded successfully")

# Define test inputs matching your NEW simplified training format
TEST_INPUTS = [
    "Problem at 4754 Amigo: The Wi-Fi is very slow. How can I fix it?",
    "Problem at 4754 Amigo: YouTube is not working on the family room TV. How do I fix it?", 
    "Problem at 4754 Amigo: The gas stove won't ignite. What should I do?",
    "Problem at 4754 Amigo: The kitchen outlets near the sink aren't working. What should I do?",
    "Problem at 4754 Amigo: The front gate won't open. What steps should I follow to fix it?",
]

def generate_comparative(model, input_text, model_name="Model"):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    
    print(f"\n{'='*60}")
    print(f"{model_name} - Input: {input_text[:60]}...")
    print(f"{'='*60}")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.1,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    
    # Check for specific terms we expect
    specific_terms = ["fast.com", "utility closet", "orange power button", "150 Mbps", "garage", "breaker panel", "Savant app"]
    found_terms = [term for term in specific_terms if term.lower() in response.lower()]
    if found_terms:
        print(f"✅ Found specific terms: {found_terms}")
    else:
        print("❌ No specific terms found - generic response")
    
    return response

# Compare outputs for each test input
for i, test_input in enumerate(TEST_INPUTS):
    print(f"\n{'#'*80}")
    print(f"TEST {i+1}: {test_input}")
    print(f"{'#'*80}")
    
    base_response = generate_comparative(base_model, test_input, "BASE MODEL")
    lora_response = generate_comparative(lora_model, test_input, "LORA MODEL")
    
    # Compare responses
    print(f"\n🔍 COMPARISON:")
    if "4754 Amigo" in lora_response and "4754 Amigo" not in base_response:
        print("✅ LoRA learned location format")
    elif "4754 Amigo" in lora_response and "4754 Amigo" in base_response:
        print("⚠️ Both mention location - checking specificity...")
    else:
        print("❌ LoRA didn't learn location format")

# Additional diagnostic: Check if LoRA weights are actually different
print("\n\n=== LoRA Weight Analysis ===")
lora_state = lora_model.state_dict()
lora_layers = [k for k in lora_state.keys() if 'lora' in k and 'weight' in k]
print(f"Found {len(lora_layers)} LoRA weight layers")

# Sample a few weights to check their magnitudes
for layer_name in lora_layers[:5]:
    weight = lora_state[layer_name]
    print(f"{layer_name}: mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")

# Test the key Wi-Fi procedure specifically
print("\n\n=== Key Procedure Test: Wi-Fi ===")
wifi_prompt = "Problem at 4754 Amigo: Internet speed is really slow. How can I fix this?"

print("🔍 BASE MODEL:")
base_inputs = tokenizer(wifi_prompt, return_tensors="pt", truncation=True).to(DEVICE)
with torch.no_grad():
    base_outputs = base_model.generate(
        input_ids=base_inputs["input_ids"],
        attention_mask=base_inputs["attention_mask"],
        max_new_tokens=200,
        do_sample=False,
        num_beams=4
    )
base_wifi_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
print(base_wifi_response)

print("\n🎯 LORA MODEL:")
lora_inputs = tokenizer(wifi_prompt, return_tensors="pt", truncation=True).to(DEVICE)
with torch.no_grad():
    lora_outputs = lora_model.generate(
        input_ids=lora_inputs["input_ids"],
        attention_mask=lora_inputs["attention_mask"],
        max_new_tokens=200,
        do_sample=False,
        num_beams=4
    )
lora_wifi_response = tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
print(lora_wifi_response)

# Check for memorized procedure
key_wifi_terms = ["fast.com", "150 Mbps", "utility closet", "laundry room", "orange power button", "Luxul router"]
found_wifi_terms = [term for term in key_wifi_terms if term in lora_wifi_response]

print(f"\n📊 MEMORIZATION CHECK:")
print(f"Expected terms: {key_wifi_terms}")
print(f"Found terms: {found_wifi_terms}")
print(f"Memorization score: {len(found_wifi_terms)}/{len(key_wifi_terms)} ({len(found_wifi_terms)/len(key_wifi_terms)*100:.1f}%)")

if len(found_wifi_terms) >= 3:
    print("🎉 SUCCESS: LoRA model memorized specific procedures!")
elif len(found_wifi_terms) >= 1:
    print("🔶 PARTIAL: LoRA model learned some specifics")
else:
    print("❌ FAILED: LoRA model still giving generic advice")