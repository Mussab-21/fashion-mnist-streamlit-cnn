import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Path to saved weights
MODEL_PATH = "best_model.pth"

def load_model(num_classes):
    # Load a ResNet (change to resnet18, resnet34, etc. if needed)
    model = models.resnet18(pretrained=False)
    
    # Replace final layer with correct number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Load trained weights
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Match ResNet input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def run_inference(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)
    return predicted.item()

if __name__ == "__main__":
    NUM_CLASSES = 2  # 🔹 Change this to match your dataset
    model = load_model(NUM_CLASSES)
    print("Model loaded successfully!")

    # Example inference
    test_image = "test.jpg"  # 🔹 Replace with your image path
    input_tensor = preprocess_image(test_image)
    prediction = run_inference(model, input_tensor)
    print("Predicted class:", prediction)
