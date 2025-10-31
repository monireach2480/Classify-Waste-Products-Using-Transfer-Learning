# Waste Classification Using Transfer Learning

## Project Overview
This project develops an automated waste classification system using transfer learning with the VGG16 model to distinguish between recyclable and organic waste. The solution addresses EcoClean's need for efficient and scalable waste sorting automation.

## Aim
Develop an automated waste classification model that accurately differentiates between recyclable and organic waste based on images, reducing manual sorting errors and improving efficiency.

## Dataset
- **Source**: Waste Classification Dataset (O-vs-R split)
- **Classes**: 2 (Organic, Recyclable)
- **Training Images**: 800
- **Validation Images**: 200
- **Test Images**: 200
- **Image Size**: 150x150 pixels

## Project Structure
```
o-vs-r-split/
├── train/
│   ├── O/          # Organic waste images
│   └── R/          # Recyclable waste images
└── test/
    ├── O/          # Test organic waste images
    └── R/          # Test recyclable waste images
```

## Technologies Used
- **Python 3.x**
- **TensorFlow 2.17.0** - Deep learning framework
- **Keras** - High-level neural networks API
- **NumPy 1.26.0** - Numerical computing
- **scikit-learn 1.5.1** - Machine learning metrics
- **Matplotlib 3.9.2** - Data visualization
- **VGG16** - Pre-trained model (ImageNet weights)

## Model Architecture

### Base Model: VGG16
- Pre-trained on ImageNet
- Frozen convolutional layers
- Feature extraction approach

### Custom Top Layers
```
- Flatten Layer
- Dense(512, activation='relu')
- Dropout(0.3)
- Dense(512, activation='relu')
- Dropout(0.3)
- Dense(1, activation='sigmoid')
```

## Training Configuration
- **Batch Size**: 32
- **Epochs**: 10
- **Validation Split**: 20%
- **Optimizer**: RMSprop (learning_rate=1e-4)
- **Loss Function**: Binary Crossentropy
- **Callbacks**: Early Stopping, Model Checkpoint, Learning Rate Scheduler

## Data Augmentation
```python
- Rescaling: 1.0/255.0
- Width Shift: 10%
- Height Shift: 10%
- Horizontal Flip: True
```

## Models Implemented

### 1. Extract Features Model
- All VGG16 layers frozen
- Only custom top layers trained
- **Test Accuracy**: ~81%

### 2. Fine-Tuned Model
- Last convolutional block (block5_conv3) unfrozen
- Fine-tuning for improved performance
- **Test Accuracy**: ~83%

## Results

### Extract Features Model
```
              precision    recall  f1-score   support
           O       0.78      0.86      0.82        50
           R       0.84      0.76      0.80        50
    accuracy                           0.81       100
```

### Fine-Tuned Model
```
              precision    recall  f1-score   support
           O       0.79      0.90      0.84        50
           R       0.88      0.76      0.82        50
    accuracy                           0.83       100
```

## Installation

1. Clone the repository
```bash
git clone <repository-url>
cd waste-classification
```

2. Install required libraries
```bash
pip install tensorflow==2.17.0
pip install numpy==1.26.0
pip install scikit-learn==1.5.1
pip install matplotlib==3.9.2
```

## Usage

1. **Download and prepare dataset**
```python
python download_data.py
```

2. **Train the model**
```python
python train_model.py
```

3. **Evaluate the model**
```python
python evaluate_model.py
```

4. **Make predictions**
```python
python predict.py --image_path <path_to_image>
```

## Key Features
- ✅ Transfer learning with VGG16
- ✅ Data augmentation for better generalization
- ✅ Early stopping to prevent overfitting
- ✅ Model checkpointing to save best weights
- ✅ Learning rate scheduling
- ✅ Comprehensive evaluation metrics
- ✅ Visualization of training progress

## Learning Objectives Achieved
1. ✅ Applied transfer learning using VGG16 for image classification
2. ✅ Prepared and preprocessed image data for machine learning
3. ✅ Fine-tuned pre-trained model to improve accuracy
4. ✅ Evaluated model performance using classification metrics
5. ✅ Visualized model predictions on test data

## Tasks Completed
- [x] Task 1: Print TensorFlow version
- [x] Task 2: Create test_generator
- [x] Task 3: Print train_generator length
- [x] Task 4: Print model summary
- [x] Task 5: Compile the model
- [x] Task 6: Plot accuracy curves (extract features model)
- [x] Task 7: Plot loss curves (fine-tune model)
- [x] Task 8: Plot accuracy curves (fine-tune model)
- [x] Task 9: Plot test image (extract features model)
- [x] Task 10: Plot test image (fine-tuned model)

## Model Files
- `O_R_tlearn_vgg16.keras` - Extract features model
- `O_R_tlearn_fine_tune_vgg16.keras` - Fine-tuned model

## Future Improvements
- [ ] Experiment with other pre-trained models (ResNet, Inception)
- [ ] Implement multi-class classification for more waste categories
- [ ] Deploy model as REST API
- [ ] Create web interface for real-time predictions
- [ ] Optimize model for edge devices

## Real-World Applications
- Municipal waste sorting facilities
- Industrial waste management
- Smart bins with automated sorting
- Recycling centers optimization
- Environmental monitoring systems

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Authors
- Len Monireach

## Acknowledgments
- VGG16 model from Keras Applications
- ImageNet dataset for pre-trained weights
- Skills Network for project guidance

## Contact
For questions or feedback, please contact [your-email@example.com]

---
**Note**: Training results may vary due to the stochastic nature of neural networks. The general trend should show decreasing loss and increasing accuracy.