# 🤖 Model Information

## Download
**Google Drive**: [best_model.pth (56MB)](https://drive.google.com/file/d/1yPQvs6tOAvhQKbpGJYgO7JCItQJCMRbw/view?usp=sharing)

## Model Architecture

### EfficientNet-B0 Backbone
```
Input: 224x224x3 RGB Image
    ↓
EfficientNet-B0 Feature Extractor
    ↓ (1280 features)
Custom Classifier:
├── Dropout(0.3)
├── Linear(1280 → 512) + BatchNorm + ReLU
├── Dropout(0.2)  
├── Linear(512 → 256) + BatchNorm + ReLU
├── Dropout(0.1)
└── Linear(256 → 59) [Output Classes]
```

## Training Details

### Dataset
- **Total Images**: 13,971
- **Classes**: 59 Indian traffic signs
- **Split**: 70% train, 15% val, 15% test
- **Augmentation**: Rotation, ColorJitter, Horizontal Flip

### Training Configuration
```python
{
    "architecture": "EfficientNet-B0",
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 25,
    "scheduler": "ReduceLROnPlateau",
    "device": "CPU"
}
```

### Performance Metrics
| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **92.25%** |
| Final Training Accuracy | 93.06% |
| Training Time | 5.8 hours |
| Total Parameters | 4,693,687 |
| Model Size | 56.8 MB |
| Inference Speed | ~165ms per image |

## Training Progress
- **Epoch 1**: 86.17% validation accuracy
- **Epoch 14**: 92.01% validation accuracy  
- **Epoch 21**: 92.25% validation accuracy (best)
- **Convergence**: Excellent with minimal overfitting

## Model Comparison
| Model | Accuracy | Parameters |
|-------|----------|------------|
| **Our EfficientNet-B0** | **92.25%** | 4.69M |
| ResNet-50 | 88.5% | 25.6M |
| MobileNet-V3 | 86.2% | 3.2M |
| VGG-16 | 84.1% | 138M |

## Usage Example

### Loading the Model
```python
import torch
from models.model import SimpleEfficientNet

# Initialize model
model = SimpleEfficientNet(num_classes=59)

# Load trained weights
checkpoint = torch.load('models/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded with {checkpoint.get('val_acc', 'N/A')}% accuracy")
```

### Inference Example
```python
from PIL import Image
from torchvision import transforms

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open('traffic_sign.jpg')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, predicted = torch.max(probabilities, 0)
    
print(f"Predicted class: {predicted.item()}")
print(f"Confidence: {confidence.item()*100:.2f}%")
```

## File Structure
```
models/
├── best_model.pth          # Main trained model (56MB)
└── model_architecture.py   # Model definition
```

## Verification
- **MD5 Hash**: `[You can add this for verification]`
- **File Size**: 56,797,703 bytes
- **PyTorch Version**: 2.7.0+cpu
- **Python Version**: 3.13

## License
This trained model is released under the same license as the project (MIT License).

---
**Note**: This model is trained specifically for Indian traffic signs and may not generalize well to other regions without fine-tuning.
