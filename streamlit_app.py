import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt

# Define the CNN model architecture (must match the trained model)
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

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define the preprocess_image function
def preprocess_image(image_input):
    """
    Loads and preprocesses an image for model inference.

    Args:
        image_input: UploadedFile object or image URL string.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    try:
        if isinstance(image_input, str) and (image_input.startswith('http') or image_input.startswith('https')):
            # Load image from URL
            response = requests.get(image_input, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            img = Image.open(response.raw).convert('L') # Convert to grayscale
        elif hasattr(image_input, 'read'):
            # Load image from uploaded file
            img = Image.open(image_input).convert('L') # Convert to grayscale
        else:
            return None

        # Resize to 28x28
        img = img.resize((28, 28))

        # Convert to PyTorch tensor and normalize
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32)
        img_tensor = img_tensor / 255.0

        # Add channel dimension and batch dimension
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

        return img_tensor, img # Return both tensor and PIL Image for display

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

# Define the predict_image function
def predict_image(image_tensor, model, device):
    """
    Performs inference on a preprocessed image tensor using the trained model.

    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor.
        model (torch.nn.Module): Trained CNN model.
        device (torch.device): Device to perform inference on.

    Returns:
        int: Predicted class index.
    """
    if image_tensor is None:
        return None

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model state dictionary
@st.cache_resource # Cache the model loading
def load_model(model_path="fashion_cnn_state.pth"):
    model = FashionCNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return model
    except FileNotFoundError:
        st.error(f"Model state dictionary not found at {model_path}. Please ensure the file exists.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.title("Fashion MNIST Image Classifier")

st.write("Upload an image of a fashion item or provide an image URL to get a prediction.")

# Image input options
image_source = st.radio("Select image source:", ("Upload Image", "Image URL"))

image_input = None
if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image_input = uploaded_file
elif image_source == "Image URL":
    image_url = st.text_input("Enter Image URL:")
    if image_url:
        image_input = image_url

if image_input is not None and model is not None:
    st.subheader("Input Image:")
    # Display the uploaded or fetched image
    if image_source == "Upload Image":
        # Need to reopen the file for display after it's been read by PIL in preprocess_image
        uploaded_file.seek(0)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    elif image_source == "Image URL":
         try:
             response = requests.get(image_input, stream=True)
             response.raise_for_status()
             img_display = Image.open(response.raw)
             st.image(img_display, caption="Image from URL", use_column_width=True)
         except Exception as e:
             st.error(f"Could not display image from URL: {e}")


    preprocessed_img_tensor, _ = preprocess_image(image_input)

    if preprocessed_img_tensor is not None:
        predicted_class_index = predict_image(preprocessed_img_tensor, model, device)

        if predicted_class_index is not None:
            predicted_class_name = class_names[predicted_class_index]
            st.subheader("Prediction:")
            st.write(f"The model predicts this is a: **{predicted_class_name}**")
        else:
            st.write("Could not get prediction.")
    else:
        st.write("Could not preprocess image for prediction.")

st.markdown("""
### Deployment Steps Explained:

1.  **Install Streamlit and other libraries:**
    Open your terminal or command prompt and run:
