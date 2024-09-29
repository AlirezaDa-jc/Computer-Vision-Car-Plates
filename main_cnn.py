import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint

from regressor import build_model
from loader import load,resize,augment
from evaluate import corners

# Constants
DATASET_DIR = 'cnn_dataset'
MODEL_SAVE_PATH = 'model.keras'
IMAGE_SIZE = (128, 128)

def main():
    # Load dataset
    X, y = load(DATASET_DIR)
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Build model
    model = build_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Model checkpoints
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min')

    # Train model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])
    
    # Predict corners on test data
    y_pred = []
    for img in X_test:
        pred_corners = corners(img, model)
        y_pred.append(pred_corners)
    y_pred = np.array(y_pred).reshape(-1, 8)
    
    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

if __name__ == '__main__':
    main()
