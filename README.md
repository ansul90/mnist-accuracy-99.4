# MNIST CNN Model Optimization

This project demonstrates the optimization of a CNN model for MNIST digit classification, achieving **99.42% test accuracy** with **less than 20,000 parameters** in **under 20 epochs**.

## 🎯 Target Requirements

- ✅ **99.4% validation/test accuracy**
- ✅ **Less than 20,000 parameters**
- ✅ **Less than 20 epochs**
- ✅ **Use of Batch Normalization**
- ✅ **Use of Dropout**
- ✅ **Use of Global Average Pooling (GAP)**

## 📊 Results Summary

| Metric | Original Model | Optimized Model | Target |
|--------|---------------|-----------------|---------|
| **Test Accuracy** | 28% | **99.42%** | 99.4% |
| **Parameters** | ~2.3M | **18,962** | <20,000 |
| **Epochs** | 1 | **12** | <20 |
| **Batch Normalization** | ❌ | ✅ | Required |
| **Dropout** | ❌ | ✅ | Required |
| **GAP** | ❌ | ✅ | Optional |

## 📋 Detailed Changes

### 1. **Total Parameter Count Test**

#### Original Model (`assignment.ipynb`):
```python
# Massive parameter count due to inefficient architecture
Conv2d(1, 32, 3, padding=1)     # 1 → 32
Conv2d(32, 64, 3, padding=1)    # 32 → 64
Conv2d(64, 128, 3, padding=1)   # 64 → 128
Conv2d(128, 256, 3, padding=1)  # 128 → 256
Conv2d(256, 512, 3)             # 256 → 512
Conv2d(512, 1024, 3)            # 512 → 1024
Conv2d(1024, 10, 3)             # 1024 → 10
# Estimated: ~2.3M parameters
```

#### Optimized Model (`train_optimized.py`):
```python
# Efficient channel progression
Conv2d(1, 10, 3, padding=1, bias=False)   # 1 → 10
Conv2d(10, 18, 3, padding=1, bias=False)  # 10 → 18
Conv2d(18, 18, 3, padding=1, bias=False)  # 18 → 18
Conv2d(18, 28, 3, padding=1, bias=False)  # 18 → 28
Conv2d(28, 28, 3, padding=1, bias=False)  # 28 → 28
Conv2d(28, 10, 3, padding=1, bias=False)  # 28 → 10
# Total: 18,962 parameters
```

**Key Optimizations:**
- Removed bias terms (`bias=False`) since Batch Normalization handles bias
- Strategic channel progression: 1→10→18→18→28→28→10
- Eliminated dense layers by using Global Average Pooling

### 2. **Use of Batch Normalization**

#### Original Model:
```python
# No Batch Normalization
def forward(self, x):
    x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
    x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
    # ...
```

#### Optimized Model:
```python
# Batch Normalization after each convolution
self.bn1 = nn.BatchNorm2d(10)
self.bn2 = nn.BatchNorm2d(18)
self.bn3 = nn.BatchNorm2d(18)
self.bn4 = nn.BatchNorm2d(28)
self.bn5 = nn.BatchNorm2d(28)
self.bn6 = nn.BatchNorm2d(10)

def forward(self, x):
    # BN applied after each conv, before activation
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    # ...
```

**Benefits:**
- Faster convergence and training stability
- Enables higher learning rates
- Reduces internal covariate shift
- Acts as regularization

### 3. **Use of Dropout**

#### Original Model:
```python
# No Dropout - prone to overfitting
```

#### Optimized Model:
```python
# Strategic Dropout placement
self.dropout1 = nn.Dropout(0.05)  # After first block
self.dropout2 = nn.Dropout(0.05)  # After second block

def forward(self, x):
    # First Block
    x = self.pool1(self.dropout1(F.relu(self.bn2(self.conv2(...)))))
    
    # Second Block  
    x = self.pool2(self.dropout2(F.relu(self.bn4(self.conv4(...)))))
```

**Implementation Details:**
- **Dropout Rate**: 0.05 (5%) - Light dropout to prevent overfitting without hurting performance
- **Placement**: After ReLU activation, before pooling operations
- **Strategic Positioning**: Only in first two blocks, not in final classification block

### 4. **Use of Global Average Pooling (GAP)**

#### Original Model:
```python
# Dense/Fully Connected final layer (implicit)
def forward(self, x):
    # ... convolutions ...
    x = x.view(-1, 10)  # Direct reshape - no proper dimensionality reduction
    return F.log_softmax(x)  # Missing dim parameter
```

#### Optimized Model:
```python
# Global Average Pooling instead of Dense layers
self.gap = nn.AdaptiveAvgPool2d(1)  # 7x7x10 → 1x1x10

def forward(self, x):
    # ... convolutions ...
    x = F.relu(self.bn6(self.conv6(...)))  # 7x7x10
    
    # Global Average Pooling
    x = self.gap(x)      # 7x7x10 → 1x1x10
    x = x.view(-1, 10)   # 1x1x10 → 10
    
    return F.log_softmax(x, dim=1)  # Fixed: added dim parameter
```

**Advantages of GAP:**
- **Massive Parameter Reduction**: Eliminates need for dense layers
- **Overfitting Prevention**: No learnable parameters in final layer
- **Translation Invariance**: Spatial averaging provides robustness
- **Architectural Elegance**: Direct mapping from feature maps to classes

## 🏗️ Architecture Comparison

### Original Architecture
```
Input(28x28x1) → Conv1(32) → Conv2(64) → Pool → Conv3(128) → Conv4(256) → Pool → Conv5(512) → Conv6(1024) → Conv7(10) → Reshape → LogSoftmax
```

### Optimized Architecture
```
Input(28x28x1) 
    ↓
Block1: Conv1(10) → BN → ReLU → Conv2(18) → BN → ReLU → Dropout(0.05) → MaxPool
    ↓
Block2: Conv3(18) → BN → ReLU → Conv4(28) → BN → ReLU → Dropout(0.05) → MaxPool  
    ↓
Block3: Conv5(28) → BN → ReLU → Conv6(10) → BN → ReLU
    ↓
GAP → LogSoftmax(dim=1)
```

## 🚀 Training Optimizations

### Data Augmentation
```python
# Original: Basic normalization only
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Optimized: Strategic augmentation
transforms.Compose([
    transforms.RandomRotation((-7.0, 7.0), fill=(0,)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### Learning Rate Scheduling
```python
# Original: Simple SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Optimized: OneCycleLR with cosine annealing
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, epochs=15, 
    steps_per_epoch=len(train_loader), 
    pct_start=0.2, anneal_strategy='cos'
)
```

### Batch Size Optimization
```python
# Original: batch_size = 128
# Optimized: batch_size = 512  # Faster training, better GPU utilization
```

## 📈 Receptive Field Analysis

| Layer | Input Size | Output Size | Receptive Field | Channels |
|-------|------------|-------------|-----------------|----------|
| Conv1 | 28×28×1 | 28×28×10 | 3×3 | 1→10 |
| Conv2 | 28×28×10 | 28×28×18 | 5×5 | 10→18 |
| Pool1 | 28×28×18 | 14×14×18 | 6×6 | 18 |
| Conv3 | 14×14×18 | 14×14×18 | 10×10 | 18→18 |
| Conv4 | 14×14×18 | 14×14×28 | 14×14 | 18→28 |
| Pool2 | 14×14×28 | 7×7×28 | 16×16 | 28 |
| Conv5 | 7×7×28 | 7×7×28 | 24×24 | 28→28 |
| Conv6 | 7×7×28 | 7×7×10 | 32×32 | 28→10 |
| GAP | 7×7×10 | 1×1×10 | - | 10 |

**Final Receptive Field**: 32×32 (larger than input 28×28) ✅

## 🛠️ Implementation Steps

### Step 1: Architecture Redesign
1. Replace massive channel progression with efficient one
2. Remove bias terms from convolutions
3. Add Batch Normalization after each convolution
4. Add strategic Dropout placement
5. Replace dense layers with Global Average Pooling

### Step 2: Training Enhancements
1. Implement data augmentation (rotation + translation)
2. Use OneCycleLR scheduler for better convergence
3. Increase batch size for efficiency
4. Add weight decay for regularization

### Step 3: Hyperparameter Tuning
1. Optimize dropout rate (0.05 found optimal)
2. Configure OneCycleLR parameters
3. Set appropriate weight decay
4. Fine-tune augmentation parameters

## 🎉 Final Results

```
🎉 Target accuracy of 99.4% achieved!
Final accuracy: 99.42%
Parameters used: 18,962 (< 20,000 ✓)
Epochs used: 12 (< 20 ✓)

📊 Model Summary:
✅ Parameters: 18,962 < 20,000
✅ Epochs: 12 < 20
✅ Batch Normalization: Used after each conv layer
✅ Dropout: Used strategically (0.05 rate)
✅ Global Average Pooling: Used instead of FC layers
✅ Data Augmentation: Rotation + translation
✅ Advanced Scheduler: OneCycleLR with cosine annealing
```

## 🔍 Key Learnings

1. **Parameter Efficiency**: Dramatic reduction from ~2.3M to 18,962 parameters while improving accuracy
2. **Batch Normalization Impact**: Essential for training stability and faster convergence
3. **Dropout Strategy**: Light dropout (0.05) prevents overfitting without hurting performance
4. **GAP Benefits**: Eliminates parameters while providing better generalization
5. **Modern Training**: OneCycleLR and data augmentation crucial for achieving high accuracy
6. **Architecture Design**: Systematic receptive field growth and efficient channel progression

This optimization demonstrates how modern deep learning techniques can achieve superior results with significantly fewer parameters through careful architectural design and training strategies.
