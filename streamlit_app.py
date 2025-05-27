"""
Traffic Sign Recognition Streamlit App
Showcases your 92.25% accuracy model
"""

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
from pathlib import Path
import time

# Page config
st.set_page_config(
    page_title="🚦 Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# Header
st.title("🚦 Indian Traffic Sign Recognition")
st.markdown("### 🏆 AI Model with 92.25% Accuracy")

# Model architecture (simplified for demo)
class SimpleEfficientNet(nn.Module):
    def __init__(self, num_classes=59):
        super().__init__()
        try:
            import timm
            self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)

            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy)
                feature_dim = features.shape[1]

            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        except ImportError:
            st.error("timm package required. Install with: pip install timm")
            st.stop()

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = Path("best_model.pth")

    if not model_path.exists():
        st.error(f"❌ Model not found at {model_path}")
        st.info("Expected model location: training_results/best_model.pth")
        st.code("Make sure your trained model file is in the correct location.")
        return None

    try:
        model = SimpleEfficientNet(num_classes=59)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_class_names():
    """Load traffic sign class names"""
    csv_path = Path(r"S:\traffic\Indian-Traffic Sign-Dataset\traffic_sign.csv")

    try:
        df = pd.read_csv(csv_path)
        return dict(zip(df['ClassId'], df['Name']))
    except:
        # Fallback class names
        return {i: f"Traffic Sign Class {i}" for i in range(59)}

def preprocess_image(image):
    """Preprocess image for prediction"""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return transform(image).unsqueeze(0)

def predict_traffic_sign(model, image, class_names):
    """Make prediction"""
    if model is None:
        return None

    try:
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # Get top 5 predictions
            top5_prob, top5_idx = torch.topk(probabilities, 5)

            predictions = []
            for i in range(5):
                class_id = top5_idx[i].item()
                confidence = top5_prob[i].item() * 100
                class_name = class_names.get(class_id, f"Class {class_id}")

                predictions.append({
                    'Rank': i + 1,
                    'Traffic Sign': class_name,
                    'Confidence (%)': round(confidence, 2)
                })

        return predictions

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    """Main app function"""

    # Sidebar
    st.sidebar.markdown("## 🤖 Model Information")
    st.sidebar.markdown("""
    **Performance Metrics:**
    - ✅ **92.25%** Validation Accuracy
    - 📊 **59** Traffic Sign Classes
    - 🔬 **EfficientNet-B0** Architecture
    - ⚙️ **4.69M** Parameters
    
    **Training Statistics:**
    - 📈 **13,971** Training Images
    - 🔄 **25** Training Epochs
    - ⏱️ **5.8** Hours Training Time
    - 🎯 **Excellent Convergence**
    """)

    # Load model and class names
    model = load_model()
    class_names = load_class_names()

    if model is None:
        st.stop()

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📸 Upload Traffic Sign Image")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of an Indian traffic sign"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Image details
            st.markdown(f"""
            **Image Information:**
            - 📏 Size: {image.size[0]} × {image.size[1]} pixels
            - 🎨 Mode: {image.mode}
            - 📁 Format: {getattr(image, 'format', 'Unknown')}
            """)

    with col2:
        if uploaded_file is not None:
            st.markdown("### 🎯 AI Prediction Results")

            with st.spinner("🔍 Analyzing traffic sign..."):
                start_time = time.time()
                predictions = predict_traffic_sign(model, image, class_names)
                prediction_time = time.time() - start_time

            if predictions:
                # Top prediction
                top_pred = predictions[0]
                confidence = top_pred['Confidence (%)']

                # Confidence color
                if confidence >= 80:
                    color = "🟢"
                    status = "High Confidence"
                elif confidence >= 60:
                    color = "🟡"
                    status = "Medium Confidence"
                else:
                    color = "🔴"
                    status = "Low Confidence"

                # Display top prediction
                st.success(f"""
                **🏆 Top Prediction:**
                
                **{top_pred['Traffic Sign']}**
                
                {color} **{confidence}%** Confidence ({status})
                
                ⚡ Prediction time: {prediction_time:.3f} seconds
                """)

                # Top 5 predictions table
                st.markdown("### 📊 All Predictions")

                df = pd.DataFrame(predictions)
                st.dataframe(df, use_container_width=True)

                # Progress bars for confidence
                st.markdown("### 📈 Confidence Levels")
                for pred in predictions[:3]:  # Show top 3
                    st.write(f"**{pred['Traffic Sign'][:40]}**")
                    st.progress(pred['Confidence (%)'] / 100.0)
                    st.write(f"{pred['Confidence (%)']}%")
                    st.write("")

        else:
            st.info("""
            👆 **Upload an image to get started!**
            
            Select a traffic sign image from your device to see the AI model in action.
            
            **Supported formats:** PNG, JPG, JPEG
            """)

    # Footer metrics
    st.markdown("---")
    st.markdown("### 🏆 Project Achievements")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Accuracy", "92.25%", "Outstanding!")

    with col2:
        st.metric("Traffic Sign Classes", "59", "Comprehensive")

    with col3:
        st.metric("Training Images", "13,971", "Robust Dataset")

    with col4:
        st.metric("Model Parameters", "4.69M", "Efficient")

if __name__ == "__main__":
    main()