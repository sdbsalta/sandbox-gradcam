import streamlit as st
import torch
import timm
import pickle
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# -------------------------------
# 1. Paths and settings
# -------------------------------
RF_MODEL_PATH = "rf_model.pkl"
SVM_MODEL_PATH = "svm_model.pkl"
MODEL_NAME = "efficientnet_b0"
IMG_SIZE = 224

# -------------------------------
# 2. Load models
# -------------------------------
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature extractor (EfficientNet-B0 backbone)
    feature_extractor = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=0,
        global_pool="avg"
    ).to(device)
    feature_extractor.eval()

    # Random Forest model + class names
    with open(RF_MODEL_PATH, "rb") as f:
        rf_data = pickle.load(f)
        rf_model = rf_data["model"]
        class_names = rf_data["classes"]

    # Scaler
    with open(SVM_MODEL_PATH, "rb") as f:
        svm_data = pickle.load(f)
        scaler = svm_data["scaler"]

    return feature_extractor, rf_model, scaler, class_names, device

feature_extractor, rf_model, scaler, class_names, device = load_models()

# -------------------------------
# 3. Preprocessing
# -------------------------------
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------------
# 4. Streamlit UI
# -------------------------------
st.title("Bone Fragment Classifier")
st.write("Upload a bone fragment image to classify it.")

img_input = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -------------------------------
# 5. Grad-CAM + RF inference
# -------------------------------
if img_input:
    pil_image = Image.open(img_input).convert("RGB")
    st.image(pil_image, caption="Input Image", use_container_width=True)

    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    # For Grad-CAM display (float RGB)
    img_rgb = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    with torch.no_grad():
        features = feature_extractor(input_tensor)
        features_np = features.cpu().numpy()
        scaled_features = scaler.transform(features_np)
        pred_idx = rf_model.predict(scaled_features)[0]
        probs = rf_model.predict_proba(scaled_features)[0]

    predicted_label = class_names[pred_idx]
    confidence = probs[pred_idx]

    st.subheader("Prediction")
    st.write(f"**{predicted_label}**")

    # -------------------------------
    # 6. Grad-CAM Visualization
    # -------------------------------
    try:
        target_layers = [feature_extractor.conv_head]
        cam = GradCAM(model=feature_extractor, target_layers=target_layers)

        # pick pseudo-target as predicted index for visualization
        cam_targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=cam_targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
        st.subheader("Grad-CAM Heatmap")
        st.image(visualization)
    except Exception as e:
        st.warning(f"Grad-CAM visualization skipped: {e}")

    st.success("Prediction complete.")
else:
    st.info("Please upload an image to start classification.")

# -------------------------------
# 7. Model Evaluation (Accuracy, Precision, Recall, F1)
# -------------------------------

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Created by Maria Rafaela V. Pelagio, Sophia Danielle B. Salta, Veneza Vielle V. Vergara</p>",
    unsafe_allow_html=True
)