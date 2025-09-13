# app.py
import io
import os
import requests

import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Page config ----------
st.set_page_config(page_title="Fashion-MNIST Classifier", layout="centered")

# --------- Model definition (must match training) ----------
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu3(out)

        out = self.fc2(out)
        return out

# --------- Class names ----------
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --------- Device ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------- Load model (cached) ----------
@st.cache_resource
def load_model(model_path="fashion_cnn_state.pth"):
    """
    Loads the PyTorch model (state_dict or full model) and returns it in eval mode on CPU/GPU.
    """
    if not os.path.exists(model_path):
        st.error(f"Model file not found at `{model_path}`. Please upload it to the app folder.")
        return None

    # instantiate model architecture
    model = FashionCNN()
    try:
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict):
            # assume state_dict
            model.load_state_dict(state)
        else:
            # saved full model object
            model = state
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    model.to(device)
    model.eval()
    return model

model = load_model("fashion_cnn_state.pth")
if model is None:
    st.stop()

# --------- Preprocessing ----------
def preprocess_image(image_input):
    """
    Accepts either:
      - uploaded file-like object (streamlit UploadedFile)
      - URL string (http/https)
    Returns tuple: (tensor_of_shape_1_1_28_28, PIL_image_for_display)
    """
    try:
        # load PIL image depending on input type
        if isinstance(image_input, str) and (image_input.startswith("http://") or image_input.startswith("https://")):
            resp = requests.get(image_input, timeout=8)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert('L')  # grayscale
        elif hasattr(image_input, "read"):
            # Streamlit uploaded file
            image_input.seek(0)
            img = Image.open(image_input).convert('L')
        else:
            return None, None

        # Resize to 28x28 (Fashion-MNIST format)
        img_resized = img.resize((28, 28))

        # Convert to numpy, normalize to [0,1]
        arr = np.array(img_resized).astype(np.float32) / 255.0

        # Convert to torch tensor with shape (1,1,28,28)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)

        return tensor, img  # Return original PIL grayscale for display (if you prefer resized, return img_resized)

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

# --------- Prediction ----------
def predict_image(image_tensor, model):
    """
    Given a preprocessed tensor (1,1,28,28) and model, return (pred_label, pred_prob, all_probs)
    """
    if image_tensor is None:
        return None, None, None

    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy().squeeze()
    top_idx = int(np.argmax(probs))
    return class_names[top_idx], float(probs[top_idx]), probs

# --------- UI ----------
st.title("Fashion-MNIST Image Classifier")
st.write("Upload an image (or paste an image URL). The model expects 28×28 grayscale Fashion-MNIST style images; real photos may be misclassified due to domain differences.")

# Input mode
input_mode = st.radio("Input type", ("Upload Image", "Image URL"))

image_input = None
uploaded_file = None

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image (jpg, png)", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image_input = uploaded_file
elif input_mode == "Image URL":
    url = st.text_input("Paste image URL (http/https)")
    if url:
        image_input = url

# When image is provided, show it and predict
if image_input is not None:
    # Display image (for uploaded file re-open stream)
    st.subheader("Input Image:")
    try:
        if input_mode == "Upload Image" and uploaded_file is not None:
            uploaded_file.seek(0)
            display_img = Image.open(uploaded_file)
            st.image(display_img, caption="Uploaded Image", use_column_width=True)
        elif input_mode == "Image URL":
            resp = requests.get(image_input, timeout=8)
            resp.raise_for_status()
            display_img = Image.open(io.BytesIO(resp.content))
            st.image(display_img, caption="Image from URL", use_column_width=True)
    except Exception as e:
        st.error(f"Could not display image: {e}")

    # Preprocess and predict
    tensor, pil_img = preprocess_image(image_input)
    if tensor is None:
        st.write("Could not preprocess image for prediction.")
    else:
        pred_label, pred_prob, all_probs = predict_image(tensor, model)
        if pred_label is None:
            st.write("Could not get prediction.")
        else:
            st.subheader("Prediction")
            st.write(f"**{pred_label}**  —  Probability: **{pred_prob:.3f}**")

            # Show top-5 probabilities
            st.write("Top probabilities:")
            sorted_idx = np.argsort(all_probs)[::-1]
            for i in sorted_idx[:5]:
                st.write(f"- {class_names[i]}: {all_probs[i]:.4f}")

# --------- Helpful deployment steps (closed triple-quote properly) ----------
st.markdown("""
### Deployment Steps (quick)
1. Install dependencies locally:
```bash
pip install streamlit torch pillow requests numpy
