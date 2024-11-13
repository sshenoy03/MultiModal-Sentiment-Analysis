import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam

# Dummy data generation
np.random.seed(42)
num_samples = 1000
text_features = np.random.rand(num_samples, 300)  # 300-dimensional text features
audio_features = np.random.rand(num_samples, 50)  # 50-dimensional audio features
video_features = np.random.rand(num_samples, 100)  # 100-dimensional video features
labels = np.random.randint(0, 2, num_samples)  # Binary sentiment labels (0 or 1)

# Split data into train and test sets
text_train, text_test, audio_train, audio_test, video_train, video_test, y_train, y_test = train_test_split(
    text_features, audio_features, video_features, labels, test_size=0.2, random_state=42
)

# 1. Early Fusion
def early_fusion_model():
    input_text = Input(shape=(300,))
    input_audio = Input(shape=(50,))
    input_video = Input(shape=(100,))
    
    # Concatenate inputs at the feature level
    merged = Concatenate()([input_text, input_audio, input_video])
    
    x = Dense(128, activation='relu')(merged)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input_text, input_audio, input_video], outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and test early fusion model
model_early = early_fusion_model()
model_early.fit([text_train, audio_train, video_train], y_train, epochs=5, batch_size=32, verbose=0)
y_pred_early = model_early.predict([text_test, audio_test, video_test])
y_pred_early = (y_pred_early > 0.5).astype(int)
accuracy_early = accuracy_score(y_test, y_pred_early)

# 2. Hybrid Fusion
def hybrid_fusion_model():
    input_text = Input(shape=(300,))
    input_audio = Input(shape=(50,))
    input_video = Input(shape=(100,))
    
    # Process each modality separately
    x_text = Dense(64, activation='relu')(input_text)
    x_audio = Dense(64, activation='relu')(input_audio)
    x_video = Dense(64, activation='relu')(input_video)
    
    # Intermediate fusion of text and audio
    fused_text_audio = Concatenate()([x_text, x_audio])
    x_fused = Dense(64, activation='relu')(fused_text_audio)
    
    # Final fusion with video
    final_fusion = Concatenate()([x_fused, x_video])
    x_final = Dense(64, activation='relu')(final_fusion)
    output = Dense(1, activation='sigmoid')(x_final)
    
    model = Model(inputs=[input_text, input_audio, input_video], outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and test hybrid fusion model
model_hybrid = hybrid_fusion_model()
model_hybrid.fit([text_train, audio_train, video_train], y_train, epochs=5, batch_size=32, verbose=0)
y_pred_hybrid = model_hybrid.predict([text_test, audio_test, video_test])
y_pred_hybrid = (y_pred_hybrid > 0.5).astype(int)
accuracy_hybrid = accuracy_score(y_test, y_pred_hybrid)

# 3. Unimodal Fusion (Ensemble)
def unimodal_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Train unimodal models
model_text = unimodal_model(300)
model_text.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model_text.fit(text_train, y_train, epochs=5, batch_size=32, verbose=0)

model_audio = unimodal_model(50)
model_audio.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model_audio.fit(audio_train, y_train, epochs=5, batch_size=32, verbose=0)

model_video = unimodal_model(100)
model_video.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model_video.fit(video_train, y_train, epochs=5, batch_size=32, verbose=0)

# Predict and ensemble results (average predictions)
y_pred_text = model_text.predict(text_test)
y_pred_audio = model_audio.predict(audio_test)
y_pred_video = model_video.predict(video_test)

y_pred_unimodal = (y_pred_text + y_pred_audio + y_pred_video) / 3
y_pred_unimodal = (y_pred_unimodal > 0.5).astype(int)
accuracy_unimodal = accuracy_score(y_test, y_pred_unimodal)

# Print results
print("Early Fusion Accuracy:", accuracy_early)
print("Hybrid Fusion Accuracy:", accuracy_hybrid)
print("Unimodal Fusion Accuracy:", accuracy_unimodal)
