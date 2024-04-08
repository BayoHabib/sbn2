from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv2D, MaxPooling2D, Flatten, concatenate, Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam

def LSTM_CNN_model(input_shape_eeg, input_shape_img, num_classes):
    # EEG signal processing branch
    eeg_input = Input(shape=input_shape_eeg, name='eeg_input')
    lstm1 = LSTM(units=32, return_sequences=True)(eeg_input)
    lstm2 = LSTM(units=32, return_sequences=True)(lstm1)
    lstm3 = LSTM(units=32)(lstm2)

    # Azimuthal projection image processing branch, inspired by VGG16
    img_input = Input(shape=input_shape_img, name='image_input')
    # Block 1
    conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    #conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_1)
    pool1 = MaxPooling2D((2, 2))(conv1_1)
    Dropout(0.25)
    # Block 2
    conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    #conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_1)
    pool2 = MaxPooling2D((2, 2))(conv2_1)
    Dropout(0.25)
    # Block 3
    conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    #conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3_1)
    #conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3_2)
    pool3 = MaxPooling2D((2, 2))(conv3_1)
    
    flat = Flatten()(pool3)

    # Combine the features from both branches
    combined = concatenate([lstm3, flat])

    # Fully connected layers
    fc1 = Dense(1184, activation='relu')(combined)
    dropout = Dropout(0.5)(fc1)
    output = Dense(num_classes, activation='sigmoid')(dropout)

    # Create the model
    model = Model(inputs=[img_input,eeg_input], outputs=output)

    return model

# Example input shapes and number of classes
input_shape_eeg = (251, 19)  # 251 timesteps, 19 features for EEG
input_shape_img = (28, 28, 3)  # 28x28 RGB images for azimuthal projections
num_classes = 1  # Example: binary classification

# Create the combined model
Hybrid_model = LSTM_CNN_model(input_shape_eeg, input_shape_img, num_classes)

# Compile the model
Hybrid_model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
Hybrid_model.summary()
#