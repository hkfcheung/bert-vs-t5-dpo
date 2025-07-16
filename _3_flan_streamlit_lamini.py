import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Load base model + LoRA adapters
@st.cache_resource
def load_model():
    base_model_name = "MBZUAI/LaMini-Flan-T5-783M"
    lora_adapter_path = "./amigo_context_lora_r32/final_model" #"./lora_flan_amigo_focused/final_model"
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    device = torch.device("cpu")  # Use CPU to match training
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def generate_response(user_input, model, tokenizer, device):
    """Generate response using the trained model"""
    prompt = f"Problem at 4754 Amigo: {user_input}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,  # Deterministic generation for consistency
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part (remove the input prompt)
    if prompt in full_response:
        response = full_response.replace(prompt, "").strip()
    else:
        # Fallback: split and take the part after the prompt
        response = full_response.split("Problem at 4754 Amigo:")[-1].strip()
    
    # Clean up any remaining artifacts
    if response.startswith(":"):
        response = response[1:].strip()
    
    return response

# Load model
try:
    model, tokenizer, device = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_message = str(e)

# Streamlit UI
st.set_page_config(page_title="4754 Amigo Home Assistant", layout="centered")
st.title("🏡 4754 Amigo Home Assistant")
st.markdown("Get specific troubleshooting help for your smart home issues at 4754 Amigo.")

# Show model status
if model_loaded:
    st.success("✅ AI Assistant is ready!")
else:
    st.error(f"❌ Model loading failed: {error_message}")
    st.stop()

# Sample questions
st.markdown("### 💡 Try asking about:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- Wi-Fi is slow")
    st.markdown("- YouTube not working on TV")
    st.markdown("- Gas stove won't ignite")
with col2:
    st.markdown("- Kitchen outlets not working")
    st.markdown("- Front gate won't open")
    st.markdown("- Camera system access")

st.markdown("---")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("What's the issue you're experiencing?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing the issue..."):
            try:
                response = generate_response(prompt, model, tokenizer, device)
                
                # Check if response contains expected elements
                if "4754 Amigo" in response:
                    st.markdown(response)
                    
                    # Add some helpful indicators
                    if any(term in response.lower() for term in ["fast.com", "utility closet", "savant", "breaker", "garage"]):
                        st.success("🎯 Specific procedure provided!")
                else:
                    st.markdown(response)
                    st.warning("⚠️ Response may be generic - try rephrasing your question.")
                
            except Exception as e:
                response = f"Sorry, I encountered an error: {str(e)}"
                st.error(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with information
with st.sidebar:
    st.markdown("### 🏠 About This Assistant")
    st.markdown("This AI assistant provides specific troubleshooting instructions for **4754 Amigo** property.")
    
    st.markdown("### 🔧 Specialized Areas")
    st.markdown("- Network & Wi-Fi issues")
    st.markdown("- Smart TV & media systems") 
    st.markdown("- Kitchen appliances")
    st.markdown("- Gate & access systems")
    st.markdown("- Security cameras")
    st.markdown("- Electrical outlets")
    
    st.markdown("### 💡 Tips")
    st.markdown("- Be specific about the problem")
    st.markdown("- Mention the device/area affected")
    st.markdown("- Ask one question at a time")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("*Powered by fine-tuned T5 model with property-specific knowledge*")