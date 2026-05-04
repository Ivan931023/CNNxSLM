import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

# --- Model Definition (ResNet-34, High-Resolution 500x500) ---
class ZernikeResNet(nn.Module):
    def __init__(self, num_predict_modes=14):
        super().__init__()
        import torchvision.models as models
        self.resnet = models.resnet34(weights=None)
        # Modify first layer to accept 1 channel grayscale instead of 3 channel RGB
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify final FC layer for 14 Zernike modes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_predict_modes)
        
    def forward(self, x): 
        return self.resnet(x)

# --- Cache Model Loading ---
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = ZernikeResNet(num_predict_modes=14).to(device)
    model_path = os.path.join('flattop_cnn_outputs', 'best_resnet_finetuned.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    else:
        return None, device

# --- Zernike Labels ---
ZERNIKE_NAMES = [
    "Z2: Tilt Y", "Z3: Tilt X", "Z4: Astigmatism 45°", "Z5: Defocus",
    "Z6: Astigmatism 0°", "Z7: Trefoil Y", "Z8: Coma Y", "Z9: Coma X",
    "Z10: Trefoil X", "Z11: Quadrafoil Y", "Z12: Sec. Astig 45°", "Z13: Spherical",
    "Z14: Sec. Astig 0°", "Z15: Quadrafoil X"
]

# --- UI Setup ---
st.set_page_config(page_title="Zernike CNN Analyzer (Hyper-Precision)", page_icon="🔬", layout="wide")

st.title("🔬 Zernike Flattop Beam Analyzer — Hyper-Precision")
st.markdown("**ResNet-34 | 500×500 High-Resolution | L1 Sparsity Finetuned**")
st.markdown("Upload a distorted flattop beam image to instantly analyze its Zernike aberration modes using our hyper-precision CNN model.")

model, device = load_model()

if model is None:
    st.error("⚠️ Model weights not found! Please make sure `flattop_cnn_outputs/best_resnet_finetuned.pth` exists. Has the hyper-precision training completed yet?")
else:
    st.success(f"Hyper-Precision Model (ResNet-34, 500×500) loaded on **{device.type.upper()}** hardware acceleration.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload Flattop Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Image")
        
        # 1. 顯示原始圖片 (保留原始色彩，無論是灰階 CCD 或是 MATLAB 的 Turbo 偽色彩)
        display_image = Image.open(uploaded_file).convert('RGB')
        st.image(display_image, use_container_width=True, caption="Uploaded Image")
        
        # 2. 將圖片轉換為灰階張量供模型推論使用 (高解析度 500x500)
        model_image = Image.open(uploaded_file).convert('L')
        img_np = np.array(model_image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        img_resized = F.interpolate(img_tensor, size=(500, 500), mode='bilinear', align_corners=False)
        img_input = img_resized.to(device)
        
    with col2:
        st.subheader("CNN Predicted Zernike Aberrations")
        with st.spinner("Analyzing image with Hyper-Precision model..."):
            with torch.no_grad():
                preds = model(img_input).squeeze(0).cpu().numpy()
        
        # Create Plotly Bar Chart
        colors = ['#EF553B' if val < 0 else '#00CC96' for val in preds]
        
        fig = go.Figure(data=[
            go.Bar(
                x=ZERNIKE_NAMES, 
                y=preds,
                marker_color=colors,
                text=[f"{val:.4f}" for val in preds],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Predicted Coefficients (Z2 to Z15) — Hyper-Precision",
            yaxis_title="Coefficient Magnitude",
            xaxis_title="Zernike Modes",
            template="plotly_dark",
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # Simple flat feedback
        rms = np.sqrt(np.mean(preds**2))
        if rms < 0.2:
            st.info(f"✅ The beam is relatively flat! (RMS Aberration: {rms:.4f})")
        else:
            st.warning(f"⚠️ High aberration detected! (RMS Aberration: {rms:.4f}). The most significant mode is **{ZERNIKE_NAMES[np.argmax(np.abs(preds))]}**.")
