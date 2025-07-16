# Property Management AI Training Experimentation

conda activate dpoenv

## 🎯 Objective
Train a T5-based language model to provide specific, location-anchored troubleshooting instructions for property management, replacing generic advice with detailed, property-specific procedures.

## 📊 Data Evolution Journey

### Phase 1: Initial Exploration (50 Examples)
- **Dataset Size:** ~50 basic instruction-response pairs
- **Format:** Simple troubleshooting Q&A
- **Result:** Model learned basic format but gave generic responses
- **Issue:** Insufficient repetition for memorization

### Phase 2: Expanded Dataset (100 Examples)  
- **Dataset Size:** ~100 examples with more variety
- **Format:** Added location prefix "<<ADDRESS>>" 
- **Result:** Model started recognizing location but still generic advice
- **Issue:** Not enough specific procedural details

### Phase 3: Targeted Variations (130 Examples)
- **Dataset Size:** 130 examples with 5 variations per procedure
- **Strategy:** Multiple phrasings of same specific instructions
- **Result:** Partial success - learned location format consistently
- **Issue:** Still falling back on pre-trained generic knowledge

### Phase 4: Aggressive Memorization (190 Examples)
- **Dataset Size:** 190 examples with 10 variations per core procedure
- **Strategy:** Maximum repetition of exact technical details
- **Result:** 🎉 **Complete Success** - Perfect memorization achieved

## 🏠 Location Anchoring Strategy

### Core Innovation: Address-Specific Instructions
- **Format:** `"INTERNAL | LOCATION: <<ADDRESS>> | [Problem Description]"`
- **Response Format:** `"At <<ADDRESS>>: [Detailed Steps]"`
- **Purpose:** Anchor all responses to specific property location

### Key Procedures Memorized:
1. **Wi-Fi Issues:** fast.com → utility closet → orange power button → 150 Mbps threshold
2. **YouTube Problems:** Savant app → Media Server → Time & Date settings
3. **Gas Stove:** Garage breaker panel → "Kitchen Stove" breaker reset
4. **Kitchen Outlets:** Backsplash vs kitchen island isolation → breaker reset
5. **Gate Issues:** Physical inspection → control arm box → garage breaker

## ⚙️ Technical Configuration

### Model Architecture
- **Base Model:** MBZUAI/LaMini-Flan-T5-783M (783M parameters)
- **Training Method:** LoRA (Low-Rank Adaptation)
- **Trainable Parameters:** 24M out of 808M total (3.06%)

### Base Model Selection Justification

**Why LaMini-Flan-T5-783M was chosen:**

1. **Optimal Size-Performance Balance**
   - 783M parameters: Large enough for complex reasoning, small enough for local training
   - Fits comfortably in consumer GPU memory (16GB+) for fine-tuning
   - Faster inference than billion+ parameter models while maintaining quality

2. **Instruction-Following Foundation**
   - Pre-trained on instruction-following tasks (Flan dataset)
   - Strong baseline for procedural, step-by-step responses
   - Already understands command → response format structure

3. **T5 Architecture Advantages**
   - Text-to-text unified framework perfect for Q&A tasks
   - Encoder-decoder design handles variable-length responses well
   - Proven track record for fine-tuning on domain-specific tasks

4. **Practical Considerations**
   - Open source and commercially usable
   - Well-documented and stable model
   - Good community support and compatibility with PEFT/LoRA
   - Reasonable training time (5 hours vs days for larger models)

5. **LoRA Compatibility**
   - Architecture well-suited for low-rank adaptation
   - Sufficient model capacity to store specific procedural knowledge
   - Allows aggressive fine-tuning without catastrophic forgetting

### Key Training Parameters
```python
# Aggressive Training Settings
learning_rate = 3e-4          # 4x higher than conservative approach
num_train_epochs = 50         # Extended training for memorization
lora_rank = 32               # Higher capacity for memorization
lora_alpha = 64              # Stronger weight updates
batch_size = 2               # More frequent parameter updates
gradient_accumulation = 4     # Effective batch size of 8
```

### Data Strategy
- **Split:** 85% train (161 examples) / 15% eval (29 examples)
- **Repetition:** 10 variations per procedure for maximum exposure
- **Format Consistency:** Simplified prompts without special tokens

## 📈 Results & Metrics

### Training Performance
- **Initial Loss:** 3.18 → **Final Loss:** 0.10 (97% improvement)
- **Evaluation Loss:** 2.80 → 0.13 (95% improvement)
- **Training Time:** ~5 hours (1,050 steps total)
- **Breakthrough Epoch:** 9 (when specific memorization began)

### Success Indicators
- ✅ **100% Location Format Compliance:** Every response starts "At <<ADDRESS>>:"
- ✅ **Specific Technical Details:** Exact procedures instead of generic advice
- ✅ **Consistent Memorization:** All 6 procedures correctly memorized
- ✅ **No Overfitting:** Train and eval loss tracked together

## 🔍 Key Learnings

### What Worked
1. **High Learning Rate (3e-4):** Essential for overriding pre-trained knowledge
2. **Massive Repetition:** 10 variations per procedure forced memorization
3. **Location Anchoring:** Consistent address format created strong association
4. **Simplified Format:** Removing special tokens improved learning efficiency
5. **Extended Training:** 50 epochs allowed complete memorization

### What Didn't Work Initially
- **Low Data Volume:** <100 examples insufficient for specific memorization
- **Conservative Learning Rate:** 8e-5 too gentle to override pre-training
- **Generic Variations:** Need exact procedural repetition, not paraphrasing
- **Complex Formatting:** Special tokens added unnecessary complexity

## 🚀 Final Outcome

Successfully transformed a general-purpose language model into a **property-specific troubleshooting assistant** that provides:

- **Exact procedural steps** (not generic advice)
- **Location-specific details** (utility closet, garage, specific breakers)
- **Technical specifications** (IP addresses, speeds, exact hardware)
- **Consistent professional format** anchored to property address

The model now serves as a reliable digital assistant for <<ADDRESS>> property management, providing maintenance staff with precise, actionable instructions for common issues.

## 📁 Repository Structure
```
├── training_data/
│   ├── initial_50_examples.jsonl
│   ├── expanded_100_examples.jsonl
│   ├── variations_130_examples.jsonl
│   └── final_190_examples.jsonl
├── models/
│   └── lora_flan_amigo_aggressive/
├── scripts/
│   ├── train_model.py
│   └── compare_basemodel.py
└── results/
    └── training_analysis.png
```

## 💻 Hardware Specifications

### Training Environment
- **Device:** Apple M2 Pro chip
- **Memory:** 16GB unified memory
- **Operating System:** macOS Sequoia 15.5
- **Training Backend:** CPU (MPS compatibility issues with LoRA)
- **Training Duration:** ~5 hours for 50 epochs

### Performance Notes
- M2 Pro's unified memory architecture allowed efficient handling of 783M parameter model
- CPU training was stable and reliable despite longer duration compared to GPU
- 16GB memory sufficient for model loading, training data, and LoRA adapters
- No memory constraints encountered during aggressive training phase

**************************************************************************************************************************
*************

dpoenv) ethancheung@ECHEUNG-MACBOOK-PRO-AI DPO_Bert % python _4_compare_basemodel.py
Loading tokenizer...
Tokenizer vocab size: 32100
Loading base model...
Loading LoRA model...

=== LoRA Model Info ===
trainable params: 0 || all params: 807,922,688 || trainable%: 0.0000

################################################################################
TEST 1: Problem at <<ADDRESS>>: The Wi-Fi is very slow. How can I fix it?
################################################################################

============================================================
BASE MODEL - Input: Problem at <<ADDRESS>>: The Wi-Fi is very slow. How can I fix...
============================================================
Response: To fix the slow Wi-Fi at <<ADDRESS>>, you can try the following steps: 1. Restart your router and modem. 2. Check if other devices can connect to the Wi-Fi network. 3. Clear your browser cache and cookies. 4. Disable any VPN or proxy connections. 5. Update your router's firmware. 6. Contact your internet service provider for further assistance.
❌ No specific terms found - generic response

============================================================
LORA MODEL - Input: Problem at <<ADDRESS>>: The Wi-Fi is very slow. How can I fix...
============================================================
Response: At <<ADDRESS>>: 1. Visit fast.com on your device to test the Wi-Fi speed. 2. If the speed is below 150 Mbps, go to the utility closet outside the laundry room (left side). 3. Open the right door of the closet and find the second shelf. 4. Below that shelf is a power strip — locate the orange power button at the end of it. 5. Hold down the orange power button for 5 seconds to power off the equipment. 6. Wait 10 seconds, then press the power button again to turn everything back on. 7. Wait about 1 minute for the modem and router to fully reboot. 8. Return to fast.com and recheck your Wi-Fi speed. 9. This process also resets the Luxul router without needing to use the admin console.
✅ Found specific terms: ['fast.com', 'utility closet', 'orange power button', '150 Mbps']

🔍 COMPARISON:
⚠️ Both mention location - checking specificity...

################################################################################
TEST 2: Problem at <<ADDRESS>>: YouTube is not working on the family room TV. How do I fix it?
################################################################################

============================================================
BASE MODEL - Input: Problem at <<ADDRESS>>: YouTube is not working on the family ...
============================================================
Response: To fix YouTube not working on the family room TV, you can try the following steps: 1. Make sure the TV is turned on and connected to the internet. 2. Go to YouTube's website and download the video you want to watch. 3. Open the video on the TV and watch it. 4. If the video doesn't play, try restarting the TV. 5. If the problem persists, contact YouTube customer support for further assistance.
❌ No specific terms found - generic response

============================================================
LORA MODEL - Input: Problem at <<ADDRESS>>: YouTube is not working on the family ...
============================================================
Response: At <<ADDRESS>>: 1. Open the Savant app and go to Media Server. 2. Select 'Media Room TV'. 3. Press 'Home' on the Savant remote — it may show 'No Signal'. 4. If YouTube is not visible, press 'Menu'. 5. Scroll sideways in the Menu until you see 'All Settings'. 6. Navigate to 'General' > 'Time and Date'. 7. Ensure that 'Set Automatically' is enabled (this fixes certificate issues). 8. Return to Home and launch YouTube again.
✅ Found specific terms: ['Savant app']

🔍 COMPARISON:
✅ LoRA learned location format

################################################################################
TEST 3: Problem at <<ADDRESS>>: The gas stove won't ignite. What should I do?
################################################################################

============================================================
BASE MODEL - Input: Problem at <<ADDRESS>>: The gas stove won't ignite. What shou...
============================================================
Response: Check the propane tank for leaks and replace it if necessary. If the problem persists, consult a professional technician for further assistance.
❌ No specific terms found - generic response

============================================================
LORA MODEL - Input: Problem at <<ADDRESS>>: The gas stove won't ignite. What shou...
============================================================
Response: At <<ADDRESS>>: 1. Check if none of the burners are lighting and it's just making a clicking sound. 2. If so, turn off the gas stove completely. 3. Go into the garage and locate the breaker panel to the left of the freezer. 4. Open the panel and look for a tripped breaker labeled 'Kitchen Stove'. 5. Flip the breaker to the left, then to the right, and back to the left to reset it. 6. Return to the stove and turn it on to confirm it's working again.
✅ Found specific terms: ['garage', 'breaker panel']

🔍 COMPARISON:
✅ LoRA learned location format

################################################################################
TEST 4: Problem at <<ADDRESS>>: The kitchen outlets near the sink aren't working. What should I do?
################################################################################

============================================================
BASE MODEL - Input: Problem at <<ADDRESS>>: The kitchen outlets near the sink are...
============================================================
Response: Check if the circuit breaker is tripped or if there are any loose wires. If the problem persists, contact your local electrical company for assistance.
❌ No specific terms found - generic response

============================================================
LORA MODEL - Input: Problem at <<ADDRESS>>: The kitchen outlets near the sink are...
============================================================
Response: At <<ADDRESS>>: 1. Confirm that the outlets along the backsplash wall near the sink are not functioning. 2. Check the outlet on the kitchen island (closest to the front door) — if that one is working, it confirms the issue is isolated. 3. Go to the garage and locate the breaker panel. 4. Look for the breaker labeled 'Kitchen Outlets'. 5. If the switch appears tripped, reset it by flipping it to the right and then back to the left. 6. Test the sink-area outlets again.
✅ Found specific terms: ['garage', 'breaker panel']

🔍 COMPARISON:
✅ LoRA learned location format

################################################################################
TEST 5: Problem at <<ADDRESS>>: The front gate won't open. What steps should I follow to fix it?
################################################################################

============================================================
BASE MODEL - Input: Problem at <<ADDRESS>>: The front gate won't open. What steps...
============================================================
Response: To fix the front gate at <<ADDRESS>>, you should follow these steps: 1. Check if the gate latch is properly tightened. 2. If the latch is not tightened, you may need to replace it with a new one. 3. If the latch is not tightened, you may need to replace it with a new one. 4. If the latch is not tightened, you may need to replace it with a new one. 5. If the latch is not tightened, you may need to replace it with a new one. 6. If the latch is not tightened, you may need to
❌ No specific terms found - generic response

============================================================
LORA MODEL - Input: Problem at <<ADDRESS>>: The front gate won't open. What steps...
============================================================
Response: At <<ADDRESS>>: 1. Check for a visible gap beneath the bottom of the gate to ensure it's not dragging on the ground. 2. If it appears tilted, inspect the upper hinge — it may be loose. Access the hinge post and tighten the bolt. 3. If the gate isn't obstructed, open the control arm box. 4. Ensure the charging light is blinking and the system light is green. 5. Flip the switch in the bottom-left corner of the control box off and back on to restart the system. Listen for three beeps to confirm the reset. 6. Check the fuse in the garage breaker panel to ensure nothing is tripped. 7. Locate the power outlet under the control arm near the gate, unplug it for 10 seconds, then plug it back in. 8. Test the gate again after a minute.
✅ Found specific terms: ['garage', 'breaker panel']

🔍 COMPARISON:
⚠️ Both mention location - checking specificity...
