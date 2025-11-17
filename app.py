import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -------------------------------
# 1. Load pretrained model
# -------------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    state_dict = torch.load("resnet50_imagenet.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Load ImageNet class names
@st.cache_data
def load_labels():
    import urllib.request, json
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urllib.request.urlopen(url) as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

labels = load_labels()

# -------------------------------
# 2. Image preprocessing
# -------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.title("ResNet-50 ImageNet Classifier")

st.write("Upload or capture an image to see what the ResNet-50 model predicts.")

source = st.radio("Choose image source:", ["Upload", "Camera"])

if source == "Upload":
    img_input = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
else:
    img_input = st.camera_input("Take a photo")

if img_input:
    img = Image.open(img_input).convert("RGB")
    st.image(img, caption="Input Image", width="stretch")

    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top5_prob, top5_catid = torch.topk(probs, 5)

    st.subheader("Top-5 Predictions:")
    for i in range(top5_prob.size(0)):
        label = labels[top5_catid[i]]
        st.write(f"{i+1}. **{label}** — {top5_prob[i]*100:.2f}%")
