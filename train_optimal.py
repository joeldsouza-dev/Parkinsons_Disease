import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Parkinsson disease.csv')

# Drop non-numeric or identifier columns
df = df.drop(columns=['name'], errors='ignore')

# Separate features and target
X = df.drop(columns=['status']).values
y = df['status'].values

# Scale the features
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Define a function to create the model (needed for KerasClassifier)
def create_model():
    model = Sequential([
        Dense(64, kernel_regularizer=regularizers.l2(0.001), input_shape=(X_pca.shape[1],)),
        LeakyReLU(alpha=0.01),
        Dropout(0.3),
        Dense(32, kernel_regularizer=regularizers.l2(0.001)),
        LeakyReLU(alpha=0.01),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall', 'AUC']
    )
    
    return model

# Initialize KFold with the number of splits (k)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create a KerasClassifier to work with scikit-learn
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=16, verbose=1)

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# List to store cross-validation results
cv_results = []
cv_precision = []
cv_recall = []
cv_f1 = []

# Perform k-fold cross-validation
for train_index, val_index in kf.split(X_pca):
    X_train_fold, X_val_fold = X_pca[train_index], X_pca[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]
    
    # Fit the model on the current fold
    history = model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), 
                        callbacks=[early_stop, reduce_lr], verbose=0)
    
    # Get predictions for the validation set
    y_pred = model.predict(X_val_fold)
    y_pred = (y_pred > 0.5).astype(int)
    
    # Calculate metrics for this fold
    val_acc = history.history['val_accuracy'][-1]
    val_precision = precision_score(y_val_fold, y_pred)
    val_recall = recall_score(y_val_fold, y_pred)
    val_f1 = f1_score(y_val_fold, y_pred)
    
    cv_results.append(val_acc)
    cv_precision.append(val_precision)
    cv_recall.append(val_recall)
    cv_f1.append(val_f1)

# Calculate average metrics across all folds
mean_val_acc = np.mean(cv_results)
mean_precision = np.mean(cv_precision)
mean_recall = np.mean(cv_recall)
mean_f1 = np.mean(cv_f1)

print(f"\nCross-Validation Results with 2 PCA Components:")
print(f"Mean Validation Accuracy: {mean_val_acc:.3f}")
print(f"Mean Validation Precision: {mean_precision:.3f}")
print(f"Mean Validation Recall: {mean_recall:.3f}")
print(f"Mean Validation F1 Score: {mean_f1:.3f}")

# Evaluate the final model on the test set
y_pred = model.predict(X_pca)
y_pred = (y_pred > 0.5).astype(int)

test_acc = model.score(X_pca, y)
test_precision = precision_score(y, y_pred)
test_recall = recall_score(y, y_pred)
test_f1 = f1_score(y, y_pred)

print(f"\nTest Set Metrics with 2 PCA Components:")
print(f"Test Accuracy: {test_acc:.3f}")
print(f"Test Precision: {test_precision:.3f}")
print(f"Test Recall: {test_recall:.3f}")
print(f"Test F1 Score: {test_f1:.3f}")

# Print explained variance ratio
print(f"\nExplained Variance Ratio with 2 PCA Components:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"Component {i+1}: {ratio:.3f}")
print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.3f}")

# Optionally plot learning curves from one of the folds if desired
plt.figure(figsize=(12, 5))

# Plot accuracy for one fold
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss for one fold
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
