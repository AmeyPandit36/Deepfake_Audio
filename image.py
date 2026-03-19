import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Page Config
st.set_page_config(page_title="Deepfake Detection", page_icon="🔍")
st.title("🔍 AI Face Authenticator")
st.write("Upload an image to check if it is Real or AI-Generated.")

# --- 1. Load the Model Architecture ---
@st.cache_resource
def load_model():
    # Must match the architecture used in training exactly
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 1) # No Sigmoid here because we used BCEWithLogitsLoss
    )
    
    # Load your saved weights
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- 2. Image Preprocessing ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- 3. Sidebar & Upload ---
uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict button
    if st.button('Analyze Image'):
        with st.spinner('Scanning for AI artifacts...'):
            img_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(img_tensor)
                # Apply Sigmoid manually since it's not in the model
                probability = torch.sigmoid(output).item()
            
            # Display Results
            st.divider()
            if probability > 0.5:
                st.error(f"🚨 **FAKE** (Confidence: {probability*100:.2f}%)")
                st.warning("This image shows signs of AI generation.")
            else:
                st.success(f"✅ **REAL** (Confidence: {(1-probability)*100:.2f}%)")
                st.info("No significant AI artifacts detected.")
