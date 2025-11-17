import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -------------------------------
# 1. Load pretrained model
# -------------------------------
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

@st.cache_resource
def load_model():
    model = efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 9)  # if 9-class
    state_dict = torch.load("best_b3.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

model = load_model()

# Load ImageNet class names
@st.cache_data
def load_labels():
    with open("classes.txt") as f:
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
        
# -------------------------------
# 4. Grad-CAM Implementation
# -------------------------------
import matplotlib.pyplot as plt
import torchvision

if "top5_catid" in locals() and len(top5_catid) > 0:
    target_class = top5_catid[0].item()

    def generate_gradcam(model, img_tensor, target_class):
        target_layer = model.features[-1][0]
        activations, gradients = [], []

        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        # Register hooks
        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_backward_hook(backward_hook)

        # Forward + backward
        output = model(img_tensor)
        class_score = output[0, target_class]
        model.zero_grad()
        class_score.backward()

        grad = gradients[0]
        act = activations[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        handle_f.remove()
        handle_b.remove()
        return cam

    # Compute Grad-CAM
    heatmap = generate_gradcam(model, input_tensor, target_class)

    # Overlay heatmap
    img_np = np.array(img.resize((224, 224))) / 255.0
    heatmap_rgb = plt.get_cmap('jet')(heatmap)[..., :3]
    overlay = 0.5 * img_np + 0.5 * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)

    st.subheader("Grad-CAM Visualization:")
    st.image(overlay, caption=f"Grad-CAM for: {labels[target_class]}")
else:
    st.warning("Upload an image first to see the Grad-CAM heatmap.")
    
# -------------------------------
# 5. Model Evaluation (Accuracy, Precision, Recall, F1)
# -------------------------------
from torchvision import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.header("Model Evaluation")

test_folder = st.text_input("Enter the path to your test dataset folder (organized by class):", "dataset")

if st.button("Evaluate Model"):
    st.info("Evaluating model... please wait.")
    try:
        # Preprocess same as training
        test_data = datasets.ImageFolder(test_folder, transform=preprocess)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

        y_true, y_pred = [], []
        progress = st.progress(0)
        total = len(test_loader)

        with torch.no_grad():
            for imgs, labels in test_loader:
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.numpy())
                progress.progress((i + 1) / total)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc*100:.2f}%")
        col2.metric("Precision", f"{prec*100:.2f}%")
        col3.metric("Recall", f"{rec*100:.2f}%")
        col4.metric("F1 Score", f"{f1*100:.2f}%")

        st.success("✅ Evaluation complete.")
    except Exception as e:
        st.error(f"Error during evaluation: {e}")
