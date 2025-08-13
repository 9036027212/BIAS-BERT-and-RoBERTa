import streamlit as st
from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Gender Bias Detector", layout="centered")
st.title("🧠 Gender Bias Detection and Mitigation")

# ---------------------------
# Model options (use lightweight ones for performance)
# ---------------------------
model_options = {
    "BERT (base-uncased)": "bert-base-uncased",  # Full BERT works fine for you
    "RoBERTa (tiny)": "sshleifer/tiny-distilroberta-base"  # Tiny version to avoid memory issues
}

# ---------------------------
# Sidebar - Model selector
# ---------------------------
st.sidebar.title("🔧 Model Options")
model_choice = st.sidebar.selectbox("Choose a model", list(model_options.keys()))
selected_model_name = model_options[model_choice]

# ---------------------------
# Cache the model loading
# ---------------------------
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return pipeline("fill-mask", model=model, tokenizer=tokenizer)

fill_mask = load_model(selected_model_name)

# ---------------------------
# Gender term sets
# ---------------------------
male_terms = {"he", "him", "his", "man", "male", "father", "boy"}
female_terms = {"she", "her", "hers", "woman", "female", "mother", "girl"}

# ---------------------------
# Bias detection
# ---------------------------
def analyze_bias(predictions):
    male_score = sum(p['score'] for p in predictions if p['token_str'].lower() in male_terms)
    female_score = sum(p['score'] for p in predictions if p['token_str'].lower() in female_terms)
    return male_score, female_score

# ---------------------------
# Bias mitigation
# ---------------------------
def mitigate_bias(predictions):
    mitigated = []
    for p in predictions:
        adjusted_score = p['score']
        if p['token_str'].lower() in male_terms.union(female_terms):
            adjusted_score = p['score'] * 0.5  # reduce bias
        mitigated.append({**p, 'adjusted_score': adjusted_score})
    return mitigated

# ---------------------------
# User input
# ---------------------------
default_text = "The nurse said [MASK] is responsible." if "BERT" in model_choice else "The nurse said <mask> is responsible."
user_input = st.text_input("Enter a sentence with a mask token", default_text)

# ---------------------------
# Action button
# ---------------------------
if st.button("Detect & Mitigate Bias"):
    mask_token = "[MASK]" if "BERT" in model_choice else "<mask>"

    if mask_token in user_input:
        with st.spinner("Analyzing predictions..."):
            try:
                # Raw predictions
                raw_preds = fill_mask(user_input)

                # Show raw predictions
                male_score, female_score = analyze_bias(raw_preds)
                st.subheader("🔍 Raw Predictions")
                for i, pred in enumerate(raw_preds):
                    st.write(f"**{i+1}.** `{pred['token_str']}` (score: {round(pred['score'], 4)})")

                # Bias detection message
                if male_score != female_score:
                    st.warning(f"⚠️ Gender Bias Detected")
                    st.write(f"Male score: **{round(male_score, 4)}** | Female score: **{round(female_score, 4)}**")
                else:
                    st.success("✅ No obvious gender bias detected.")

                # Mitigation
                st.subheader("🛠️ Mitigated Predictions")
                mitigated_preds = mitigate_bias(raw_preds)
                for i, pred in enumerate(mitigated_preds):
                    st.write(f"**{i+1}.** `{pred['token_str']}` (adjusted score: {round(pred['adjusted_score'], 4)})")

                # Scores after mitigation
                new_male = sum(p['adjusted_score'] for p in mitigated_preds if p['token_str'].lower() in male_terms)
                new_female = sum(p['adjusted_score'] for p in mitigated_preds if p['token_str'].lower() in female_terms)

                st.markdown("**🎯 Scores After Mitigation:**")
                st.write(f"Male score: **{round(new_male, 4)}** | Female score: **{round(new_female, 4)}**")

                if abs(new_male - new_female) < 0.01:
                    st.success("✅ Bias successfully mitigated.")
                else:
                    st.warning("⚠️ Some residual bias remains.")

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning(f"Please include the mask token `{mask_token}` in your sentence.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Developed for gender bias detection and mitigation using masked language models.")
