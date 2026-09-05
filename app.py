import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import json


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="centered"
)


# -----------------------------
# Load class names
# -----------------------------
CLASS_NAMES_URL = (
    "https://huggingface.co/mdameen/plant-disease-resnet18/"
    "resolve/main/class_names.json"
)

import requests

class_names = requests.get(CLASS_NAMES_URL).json()


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="mdameen/plant-disease-resnet18",
        filename="plant_disease_resnet18.pth"
    )

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# UI
# -----------------------------
st.title("🌱 Plant Disease Classifier")

st.write(
    "Upload a plant leaf image to predict its disease "
    "using a ResNet18-based deep learning model."
)

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        top5_probabilities, top5_indices = torch.topk(
            probabilities,
            5
        )

    predicted_idx = top5_indices[0][0].item()
    confidence = top5_probabilities[0][0].item()

    predicted_class = class_names[predicted_idx]

    st.subheader("Prediction")

    st.success(
        f"🌿 {predicted_class.replace('___', ' → ')}"
    )

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )

    if confidence >= 0.95:
        st.info("✅ High confidence prediction")
    else:
        st.warning(
            "⚠️ Low-confidence prediction — "
            "the result may be uncertain."
        )

    st.subheader("Top 5 Predictions")

    for probability, index in zip(
        top5_probabilities[0],
        top5_indices[0]
    ):

        label = class_names[index.item()]
        score = probability.item() * 100

        st.write(
            f"**{label.replace('___', ' → ')}** — "
            f"{score:.2f}%"
        )

        st.progress(
            min(float(probability.item()), 1.0)
        )


# -----------------------------
# Model information
# -----------------------------
st.divider()

st.caption(
    "Model: ResNet18 with transfer learning | "
    "38 plant disease classes | "
    "Test Accuracy: 99.48%"
)
