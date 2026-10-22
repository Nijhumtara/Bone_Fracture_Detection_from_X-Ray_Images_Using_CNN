# Bone Fracture Detection from X-Ray Images Using CNN

A deep learning project that classifies bone X-ray images into **Fractured** or **Not Fractured** using **EfficientNetB0 transfer learning** with TensorFlow and Keras.

## 📌 Overview

This project explores automated bone-fracture classification using computer vision and deep learning. It includes dataset extraction, image preprocessing, stratified validation/test splitting, data augmentation, transfer learning, fine-tuning, and model evaluation.

The model uses EfficientNetB0 pretrained on ImageNet as a feature extractor, followed by a custom classification head. Training is performed in two stages: first training the classification head with the pretrained base frozen, then fine-tuning selected later layers with a lower learning rate.

This is an educational project and is **not a clinically validated diagnostic system**.

## 🛠️ Tech Stack

- Python
- TensorFlow and Keras
- EfficientNetB0
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## 📊 Dataset

- **Input:** Bone X-ray images
- **Image size:** 224 × 224 pixels
- **Input format:** RGB
- **Target classes:** `fractured` and `not fractured`
- **Dataset source:** [Bone Fracture Detection Using X-Ray Images — Kaggle](https://www.kaggle.com/datasets/devbatrax/fracture-detection-using-x-ray-images)

The notebook expects a dataset archive named `archive.zip` and searches for `train` and `val` directories after extraction. The validation directory is split into validation and test subsets using stratified sampling.

For an unbiased final evaluation, use an independent test dataset that has not been used for model selection.

## ⚙️ Workflow

1. Extract the dataset from `archive.zip`.
2. Locate the training and validation directories.
3. Load training images and establish a consistent class order.
4. Split validation images into stratified validation and test subsets.
5. Resize images to 224 × 224 pixels.
6. Apply mild data augmentation during training.
7. Load EfficientNetB0 pretrained on ImageNet.
8. Add global average pooling, dense classification, and dropout layers.
9. Train the classification head with the pretrained base frozen.
10. Fine-tune selected later EfficientNetB0 layers using a lower learning rate.
11. Plot training/validation accuracy and loss.
12. Evaluate the model with test accuracy, precision, recall, F1-score, and a confusion matrix.
13. Optionally predict the class of an individual X-ray image.
14. Save the trained model in Keras format.

## 🧠 Model Architecture and Training

**Base model:** EfficientNetB0 pretrained on ImageNet  
**Input size:** 224 × 224 × 3  
**Classification head:** Global Average Pooling → Dropout → Dense (128 units, ReLU) → Dropout → Dense (1 unit, Sigmoid)

**Training configuration:**

- **Optimizer:** Adam
- **Stage 1 learning rate:** 0.001
- **Fine-tuning learning rate:** 0.00001
- **Loss function:** Binary Cross-Entropy
- **Batch size:** 32
- **Epochs:** Up to 15 per stage
- **Regularization:** Data augmentation and dropout
- **Callbacks:** Model checkpointing, early stopping, and learning-rate reduction on plateau

The actual number of epochs completed depends on early stopping.

## 📈 Performance Analysis and Results

The final test accuracy and class-wise metrics are generated when the notebook is run. **No final accuracy is claimed in this README because the updated transfer-learning model has not yet been evaluated in a completed training run.**

The notebook reports:

- Test accuracy
- Precision, recall, and F1-score for each class
- Classification report
- Confusion matrix
- Training and validation accuracy curves
- Training and validation loss curves

Use the results from your final run when reporting performance. Accuracy alone is not sufficient to assess a medical-image classifier; examine false positives, false negatives, and per-class recall as well.

## 🖼️ Visualizations

The notebook generates and saves:

- `accuracy_curve.png` — training and validation accuracy
- `loss_curve.png` — training and validation loss
- `confusion_matrix.png` — true and predicted class counts

Run the notebook to generate the latest figures before adding them to this repository.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Nijhumtara/YOUR-REPOSITORY-NAME.git
cd YOUR-REPOSITORY-NAME
```

Replace `YOUR-REPOSITORY-NAME` with the actual repository name.

### 2. Install dependencies

```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn
```

### 3. Prepare the dataset

Download the dataset from Kaggle and upload its ZIP file as `archive.zip` to your Google Colab session. The notebook extracts it and searches for the training and validation directories.

### 4. Run the notebook

Open `Bone_Fracture_Detection_CNN_Transfer_Learning.ipynb` in Google Colab or Jupyter Notebook and run the cells in order. A GPU is recommended to speed up training.

### 5. Review results

Inspect the training curves, test metrics, classification report, and confusion matrix. Update this README with the measured results after completing the final run.

## 💾 Saved Model

After training, the notebook saves:

- `bone_fracture_detection_cnn.keras` — final trained model
- `best_fracture_cnn.keras` — checkpoint selected using validation loss

## 🔮 Future Improvements

- Evaluate on a separate, representative test dataset.
- Investigate class imbalance and classification-threshold selection using validation data only.
- Compare EfficientNetB0 with other pretrained architectures.
- Improve generalization with carefully selected augmentation and hyperparameter tuning.
- Validate performance on diverse images before considering any real-world application.

## 👤 Author

**Nazma Begum**  
GitHub: [@Nijhumtara](https://github.com/Nijhumtara)

---

**Disclaimer:** This project is for educational and research purposes only. It is not intended to diagnose fractures or replace professional medical assessment.
