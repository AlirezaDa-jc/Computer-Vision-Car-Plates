import tensorflow as tf
from sklearn.model_selection import train_test_split
from loader import load_data
from regressor import create_model

def train_model(image_dir, label_dir):
    # Load data
    images, points = load_data(image_dir, label_dir, augment_data=True)
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, points, test_size=0.2, random_state=42)
    
    # Create model
    model = create_model()
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    
    # Train model
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=32)
    
    # Save model
    model.save('license_plate_regressor.h5')
    
    return model, history

if __name__ == "__main__":
    image_dir = './images'
    label_dir = './labels'
    train_model(image_dir, label_dir)
