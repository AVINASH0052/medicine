# Health Issue Analyzer

A Streamlit-based application that analyzes health issues described by the user, suggests precautions, and provides possible medicines. The app also allows users to upload images (e.g., skin conditions) for additional context.

---

## Features

- **Health Issue Analysis**: Enter symptoms or health issues in text form, and get relevant precautions and medication suggestions.
- **Optional Image Upload**: Upload an image (e.g., skin rashes) for additional information.
- **NVIDIA API Integration**: Uses NVIDIA's advanced language models for health-related natural language understanding.
- **Fallback to Local Models**: Ensures reliability by using HuggingFace's transformers if the NVIDIA API is unavailable.

---

## Requirements

1. **Python**: Ensure you have Python 3.8+ installed.
2. **Libraries**: Install the required libraries using:
   ```bash
   pip install streamlit transformers requests pillow
