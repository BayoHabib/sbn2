from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv2D, MaxPooling2D, Flatten, concatenate, Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam

def LSTM_CNN_model2(input_shape_eeg, input_shape_img, num_classes):
    # EEG signal processing branch
    eeg_input = Input(shape=input_shape_eeg, name='eeg_input')
    lstm1 = LSTM(units=64, return_sequences=True)(eeg_input)
    bn1 = BatchNormalization()(lstm1)  # Apply BN after LSTM
    lstm2 = LSTM(units=64, return_sequences=True)(bn1)
    bn2 = BatchNormalization()(lstm2)  # Apply BN after LSTM
    lstm3 = LSTM(units=64)(bn2)
    bn3 = BatchNormalization()(lstm3)  # Apply BN after LSTM

    # Azimuthal projection image processing branch, inspired by VGG16
    img_input = Input(shape=input_shape_img, name='image_input')
    # Block 1
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
    bn4 = BatchNormalization()(conv1_1)  # Apply BN after Conv
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn4)
    bn5 = BatchNormalization()(conv1_2)  # Apply BN after Conv
    pool1 = MaxPooling2D((2, 2))(bn5)
    #Dropout(0.25)
    
    # Block 2
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    bn6 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn6)
    bn7 = BatchNormalization()(conv2_2)
    pool2 = MaxPooling2D((2, 2))(bn7)
    #Dropout(0.25)
    
    # Block 3
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    bn8 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(bn8)
    bn9 = BatchNormalization()(conv3_2)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(bn9)
    bn10 = BatchNormalization()(conv3_3)
    pool3 = MaxPooling2D((2, 2))(bn10)
    #Dropout(0.25)
    
    flat = Flatten()(pool3)

    # Combine the features from both branches
    combined = concatenate([bn3, flat])

    # Fully connected layers
    fc1 = Dense(1024, activation='relu')(combined)
    dropout1 = Dropout(0.5)(fc1)  # Dropout for regularization
    output = Dense(num_classes, activation='sigmoid')(dropout1)

    # Create the model
    model = Model(inputs=[img_input, eeg_input], outputs=output)

    return model

# Example input shapes and number of classes
input_shape_eeg = (251, 19)  # 251 timesteps, 19 features for EEG
input_shape_img = (28, 28, 3)  # 28x28 RGB images for azimuthal projections
num_classes = 1  # Example: binary classification

# Create the combined model
Hybrid_model2 = LSTM_CNN_model2(input_shape_eeg, input_shape_img, num_classes)

# Compile the model
Hybrid_model2.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
Hybrid_model2.summary()
#