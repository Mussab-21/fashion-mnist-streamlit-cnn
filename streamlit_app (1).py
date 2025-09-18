import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import requests
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Define the CNN model architecture (must match the trained model)
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2);

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2);

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

# Define the preprocess_image function (UPDATED)
def preprocess_image(image_input):
    """
    Loads, preprocesses, and resizes an image for model inference, handling arbitrary resolutions and backgrounds.

    Args:
        image_input: UploadedFile object or image URL string.

    Returns:
        torch.Tensor: Preprocessed and resized image tensor (1, 1, 28, 28).
        PIL.Image: The opened PIL Image object.
    """
    try:
        if isinstance(image_input, str) and (image_input.startswith('http') or image_input.startswith('https')):
            # Load image from URL using requests stream
            response = requests.get(image_input, stream=True) # Use stream=True
            response.raise_for_status()  # Raise an exception for bad status codes
            img = Image.open(response.raw).convert('L')  # Open directly from the raw stream
        elif hasattr(image_input, 'read'):
            # Load image from uploaded file
            img = Image.open(image_input).convert('L')  # Convert to grayscale
        else:
            return None, None

        # Maintain aspect ratio and resize to fit within 28x28 bounds
        img.thumbnail((28, 28))

        # Create a new 28x28 blank image with a black background
        new_img = Image.new("L", (28, 28), color=0)

        # Calculate paste position to center the thumbnail
        paste_x = (28 - img.width) // 2
        paste_y = (28 - img.height) // 2

        # Paste the resized image onto the black background
        new_img.paste(img, (paste_x, paste_y))

        # Convert to PyTorch tensor and normalize
        img_tensor = torch.tensor(np.array(new_img), dtype=torch.float32)
        img_tensor = img_tensor / 255.0

        # Add channel dimension and batch dimension
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

        return img_tensor, img # Return both tensor and original PIL Image

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
    try:
        if isinstance(image_input, str): # Handle URL input for display
             response = requests.get(image_input, stream=True)
             response.raise_for_status()
             img_display = Image.open(response.raw)
             st.image(img_display, caption="Image from URL", use_column_width=True)
        else: # Handle uploaded file for display
            # Need to reopen the file for display after it's potentially read by PIL in preprocess_image
            image_input.seek(0)
            st.image(image_input, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
         st.error(f"Could not display input image: {e}")


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
    ```bash
    pip install streamlit torch torchvision Pillow requests numpy matplotlib
    ```
    Make sure you have the `fashion_cnn_state.pth` file (saved in the previous step) in the same directory where you will create your Streamlit app script.

2.  **Create the Streamlit Script:**
    Save the Python code above as `streamlit_app.py` (or any other `.py` file name) in the same directory as your `fashion_cnn_state.pth` file.

3.  **Structure of the Script (`streamlit_app.py`):**
    *   **Import necessary libraries:** `streamlit`, `torch`, `torch.nn`, `PIL` for image handling, `requests` for URLs, `numpy`, and `matplotlib`.
    *   **Define the CNN Model:** The `FashionCNN` class is defined exactly as it was during training. This is necessary to load the saved `state_dict` correctly.
    *   **Define Class Names:** The `class_names` list is included to map the predicted class index to a human-readable fashion item name.
    *   **`preprocess_image` Function:** This function takes either an uploaded file object or a URL, loads the image (converting to grayscale), resizes it to 28x28 while maintaining aspect ratio and centering on a black background, converts it to a PyTorch tensor, normalizes the pixel values, and adds the necessary channel and batch dimensions. It also returns the PIL Image object for display.
    *   **`predict_image` Function:** This function takes the preprocessed image tensor and the loaded model, sets the model to evaluation mode, performs a forward pass, and returns the index of the predicted class.
    *   **Load the Model:** The `load_model` function loads the saved `state_dict` into an instance of the `FashionCNN` model. The `@st.cache_resource` decorator is used to cache the model loading, so it only happens once when the app starts.
    *   **Streamlit App Layout:**
        *   `st.title()`: Sets the title of the web application.
        *   `st.write()`: Adds introductory text.
        *   `st.radio()`: Creates radio buttons for the user to choose between uploading an image or providing a URL.
        *   `st.file_uploader()`: Provides an interface for uploading image files.
        *   `st.text_input()`: Provides a text field for entering an image URL.
        *   Conditional logic (`if image_input is not None and model is not None`): This block executes when an image is provided and the model is loaded.
        *   `st.subheader()`: Adds subheadings.
        *   `st.image()`: Displays the input image.
        *   Call `preprocess_image` and `predict_image`.
        *   Display the prediction result using `st.write()`.

4.  **Run the Streamlit Application:**
    Open your terminal or command prompt, navigate to the directory where you saved `streamlit_app.py` and `fashion_cnn_state.pth`, and run:
    ```bash
    streamlit run streamlit_app.py
    ```
    This will start a local web server, and your Streamlit application will open in your default web browser.

5.  **Using the Application:**
    *   In the web browser, select either "Upload Image" or "Image URL".
    *   If uploading, click "Choose an image..." and select your image file.
    *   If using a URL, paste the image URL into the text box.
    *   The application will display the image and the predicted fashion item class.

This explanation provides a comprehensive guide to deploying the trained PyTorch Fashion MNIST model using Streamlit, covering all the necessary steps from setup to running the application.
""")
