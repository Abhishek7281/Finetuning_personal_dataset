import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from peft import PeftModel

@st.cache_resource(show_spinner=True)
def load_model():
    base_model_id = "meta-llama/Llama-2-7b-chat-hf"
    
    nf4Config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True, add_eos_token=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=nf4Config,
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=True
    )
    
    model_finetuned = PeftModel.from_pretrained(base_model, "checkpoint-20")
    model_finetuned.eval()
    
    return tokenizer, model_finetuned

def generate_answer(tokenizer, model, question):
    prompt = f"Question: {question} Just answer this question accurately and concisely.\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main():
    st.title("LLaMA 2 Fine-tuned Q&A App")
    st.write("Ask a question related to your fine-tuned domain and get an answer.")
    
    tokenizer, model = load_model()
    
    user_question = st.text_input("Enter your question:")
    
    if st.button("Get Answer") and user_question.strip() != "":
        with st.spinner("Generating answer..."):
            answer = generate_answer(tokenizer, model, user_question)
        st.markdown("### Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()
