import streamlit as st
import requests
from transformers import pipeline
from PIL import Image

# NVIDIA API Configuration
NVIDIA_API_KEY = "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"  # Replace with actual API key
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# HuggingFace pipeline for local fallback
local_model = pipeline("text-classification", model="distilbert-base-uncased")

# Function to query NVIDIA API
def analyze_health_issue_nvidia(issue_text):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "messages": [{"role": "user", "content": f"Suggest precautions and medicines for: {issue_text}"}],
        "temperature": 0.5,
        "max_tokens": 500,
    }
    response = requests.post(NVIDIA_BASE_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        return f"Error: Unable to analyze health issue. Status code {response.status_code}"

# Streamlit app setup
st.title("Health Issue Analyzer")
st.write("Enter your health issue below, and the system will provide precautions and possible medicines.")

# Input for health issue
health_issue = st.text_input("Describe your health issue:")

# Optional image upload
uploaded_image = st.file_uploader("Upload an image (optional, e.g., skin condition):", type=["jpg", "png", "jpeg"])

# Analyze button
if st.button("Analyze"):
    if health_issue:
        with st.spinner("Analyzing..."):
            try:
                # First, attempt NVIDIA API
                result = analyze_health_issue_nvidia(health_issue)
            except Exception as e:
                # Fallback to local model if NVIDIA API fails
                result = local_model(health_issue)
                result = result[0]["label"]

        st.subheader("Analysis Result")
        st.write(result)
    else:
        st.error("Please enter a health issue to analyze.")

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

st.write("Disclaimer: This tool provides suggestions based on input text and is not a substitute for professional medical advice.")
