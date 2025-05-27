# 🚦 Indian Traffic Sign Recognition System

<div align="center">

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.28+-orange.svg)
![Accuracy](https://img.shields.io/badge/accuracy-92.25%25-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**🎯 AI-Powered Traffic Sign Recognition with 92.25% Accuracy**

*Revolutionizing road safety through advanced computer vision and deep learning*

</div>

---

## 🌟 **Project Highlights**

<table align="center">
<tr>
<td align="center">
  <img src="https://img.icons8.com/color/96/000000/artificial-intelligence.png" width="80"/>
  <br><b>🧠 Smart AI Model</b>
  <br>EfficientNet-B0 Architecture
  <br>4.69M Parameters
</td>
<td align="center">
  <img src="https://img.icons8.com/color/96/000000/speed.png" width="80"/>
  <br><b>⚡ Lightning Fast</b>
  <br>Real-time Predictions
  <br>0.111 seconds per image
</td>
<td align="center">
  <img src="https://img.icons8.com/color/96/000000/accuracy.png" width="80"/>
  <br><b>🎯 High Accuracy</b>
  <br>92.25% Validation
  <br>59 Traffic Sign Classes
</td>
<td align="center">
  <img src="https://img.icons8.com/color/96/000000/web.png" width="80"/>
  <br><b>🌐 Web Application</b>
  <br>Interactive Interface
  <br>Drag & Drop Upload
</td>
</tr>
</table>

---

## 🎮 **Live Demo - Experience the Magic**

### 🖥️ **Streamlit Web Application Interface**

<div align="center">

![Streamlit App Interface](images/streamlit_interface.png)

*🎨 Clean, intuitive interface showcasing model performance metrics and upload functionality*

</div>

### 🎯 **AI Prediction in Action**

<div align="center">

![Prediction Results](images/prediction_results.png)

*⚡ Real-time traffic sign recognition with confidence scoring and detailed analysis*

</div>

### 📊 **Confidence Analysis Dashboard**

<div align="center">

![Confidence Levels](images/confidence_levels.png)

*📈 Comprehensive prediction confidence visualization with ranked results*

</div>

---

## 🚀 **Quick Start Guide**

### 🛠️ **Installation**

```bash
# 📥 Clone the repository
git clone https://github.com/your-username/indian-traffic-sign-recognition.git
cd indian-traffic-sign-recognition

# 🐍 Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 📦 Install dependencies
pip install -r requirements.txt
```

### 🤖 **Download Pre-trained Model**

<div align="center">

| 📊 **Model Details** | 📝 **Information** |
|:---:|:---:|
| **Architecture** | EfficientNet-B0 |
| **Accuracy** | 92.25% |
| **File Size** | 56.8 MB |
| **Classes** | 59 Indian Traffic Signs |

</div>

```bash
# 📥 Download from Google Drive
https://drive.google.com/file/d/1yPQvs6tOAvhQKbpGJYgO7JCItQJCMRbw/view?usp=sharing

# 📁 Place in models directory
mkdir models
# Move downloaded file to: models/best_model.pth
```

### 🎮 **Launch Application**

```bash
# 🚀 Start Streamlit app
streamlit run streamlit_app.py

# 🌐 Open browser and navigate to:
# http://localhost:8501
```

---

## 🏆 **Model Performance**

<div align="center">

### 📈 **Training Results**

| 🎯 **Metric** | 📊 **Score** | 🎨 **Status** |
|:---:|:---:|:---:|
| **Validation Accuracy** | 92.25% | ![Excellent](https://img.shields.io/badge/-Excellent-brightgreen) |
| **Training Images** | 13,971 | ![Robust](https://img.shields.io/badge/-Robust-blue) |
| **Training Epochs** | 25 | ![Optimized](https://img.shields.io/badge/-Optimized-orange) |
| **Training Time** | 5.8 Hours | ![Efficient](https://img.shields.io/badge/-Efficient-yellow) |
| **Model Parameters** | 4.69M | ![Lightweight](https://img.shields.io/badge/-Lightweight-purple) |

</div>

---

## 🎨 **Key Features**

<div align="center">

```mermaid
graph TD
    A[📸 Upload Image] --> B[🔍 AI Processing]
    B --> C[🎯 Prediction Results]
    C --> D[📊 Confidence Score]
    C --> E[⚡ 0.111s Response Time]
    C --> F[📈 Visual Analytics]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#f1f8e9
```

</div>

### ✨ **Core Capabilities**

- 🎯 **High Accuracy Recognition**: 92.25% validation accuracy across 59 traffic sign classes
- ⚡ **Real-time Processing**: Lightning-fast predictions in just 0.111 seconds
- 🎨 **Interactive Interface**: User-friendly Streamlit web application
- 📊 **Detailed Analytics**: Confidence scoring and prediction visualization
- 🚀 **Production Ready**: Optimized EfficientNet-B0 architecture
- 📱 **Responsive Design**: Works seamlessly across devices

---

## 🛣️ **Supported Traffic Signs**

<div align="center">

| 🚫 **Prohibition** | ⚠️ **Warning** | ℹ️ **Information** | 🎯 **Mandatory** |
|:---:|:---:|:---:|:---:|
| No Entry | Speed Limit | Parking | Keep Right |
| No Overtaking | Sharp Turn | Hospital | Roundabout |
| No U-Turn | School Zone | Fuel Station | Traffic Light |

*And 47+ more traffic sign categories...*

</div>

---

## 🔧 **Technical Architecture**

<div align="center">

```
📊 Data Pipeline
├── 🖼️  Image Preprocessing
├── 🔄  Data Augmentation  
├── 🎯  EfficientNet-B0 Model
├── ⚡  Real-time Inference
└── 🎨  Streamlit Interface
```

</div>

### 🧠 **Model Architecture**

- **Base Model**: EfficientNet-B0 (Pre-trained on ImageNet)
- **Fine-tuning**: Custom classification head for Indian traffic signs
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout and data augmentation for robust training

---

## 📈 **Performance Metrics**

<div align="center">

![Training Progress](images/training_curves.png)

*📊 Training and validation accuracy progression over 25 epochs*

</div>

---

## 🤝 **Contributing**

We welcome contributions! Here's how you can help:

- 🐛 **Bug Reports**: Found an issue? Let us know!
- ✨ **Feature Requests**: Have ideas for improvements?
- 📝 **Documentation**: Help improve our docs
- 🧪 **Testing**: Add test cases and improve coverage

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 🌟 **Star this repository if you found it helpful!**

**Made with ❤️ by [Your Name]**

![GitHub stars](https://img.shields.io/github/stars/your-username/indian-traffic-sign-recognition?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/indian-traffic-sign-recognition?style=social)

</div>
