import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Concatenate, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Parametri 
MAX_SAMPLES = 20000
TOP_N_STYLES = 50 
TOP_N_ARTISTS = 40
TOP_N_GENRES = 40 
IMG_SIZE = 96
BATCH_SIZE = 32
NUM_EPOCHS = 20 
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Ucitavanje podataka i podela dataset-a
dataset = load_dataset("Artificio/WikiArt")
dataset_train_full = dataset["train"].select(range(MAX_SAMPLES))

df_full = pd.DataFrame(dataset_train_full)
style_counts = df_full['style'].value_counts()
top_styles = style_counts.head(TOP_N_STYLES).index.tolist()

filtered_indices = df_full[df_full['style'].isin(top_styles)].index.tolist()
dataset_train_filtered = dataset_train_full.select(filtered_indices)

train_split = dataset_train_filtered.train_test_split(test_size=0.3, seed=42)
train_dataset = train_split["train"]
test_valid_dataset = train_split["test"]
test_valid_split = test_valid_dataset.train_test_split(test_size=0.5, seed=42)
val_dataset = test_valid_split["train"]
test_dataset = test_valid_split["test"]

# Preprocesiranje metapodataka
df_for_ohe = pd.DataFrame(dataset_train_filtered)
top_artists = df_for_ohe['artist'].value_counts().head(TOP_N_ARTISTS).index.tolist()
top_genres = df_for_ohe['genre'].value_counts().head(TOP_N_GENRES).index.tolist()

def categorize_artist(artist_name):
    if artist_name in top_artists:
        return artist_name
    else:
        return 'Other Artist'

def categorize_genre(genre_name):
    if genre_name in top_genres:
        return genre_name
    else:
        return 'Other Genre'

def preprocess_metadata(df):
    df['date'] = pd.to_numeric(df['date'], errors='coerce') 
    date_median = df['date'].median()
    df['date'] = df['date'].fillna(date_median)

    date_scaled = df['date'].values.reshape(-1,1) / 2025

    df['artist'] = df['artist'].apply(categorize_artist) 
    df['genre'] = df['genre'].apply(categorize_genre)
    
    artist_encoded = pd.get_dummies(df['artist']).reindex(columns=top_artists + ['Other Artist'], fill_value=0)
    genre_encoded = pd.get_dummies(df['genre']).reindex(columns=top_genres + ['Other Genre'], fill_value=0)

    return np.concatenate([date_scaled, artist_encoded.values, genre_encoded.values], axis=1)

train_meta = preprocess_metadata(pd.DataFrame(train_dataset)).astype(np.float32)
val_meta = preprocess_metadata(pd.DataFrame(val_dataset)).astype(np.float32)
test_meta = preprocess_metadata(pd.DataFrame(test_dataset)).astype(np.float32)

# Pretprocesiranje labela i slika
y_train = train_dataset['style']
y_val = val_dataset['style']
y_test = test_dataset['style']

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

y_train_cat = to_categorical(y_train_enc)
y_val_cat = to_categorical(y_val_enc)
y_test_cat = to_categorical(y_test_enc)
NUM_CLASSES = y_train_cat.shape[1] 

def load_and_preprocess_images(hf_dataset):
    images = []
    for img in hf_dataset['image']:
        img_array = np.array(img)
        resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
        images.append(resized_img / 255.0) 

    return np.array(images)

X_train_img = load_and_preprocess_images(train_dataset)
X_val_img = load_and_preprocess_images(val_dataset)
X_test_img = load_and_preprocess_images(test_dataset)

# Augmentacija slika
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, 
    zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)

# Samo CNN model
def build_cnn_model(input_shape, num_classes):
    input_img = Input(shape=input_shape)

    x = Conv2D(32, (3,3), activation='relu')(input_img)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3,3), activation='relu')(x) 
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.6)(x)
    output = Dense(num_classes, activation='softmax')(x)

    cnn_model = Model(inputs=input_img, outputs=output)
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return cnn_model

cnn_model = build_cnn_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)

history_cnn = cnn_model.fit(
    datagen.flow(X_train_img, y_train_cat, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train_img)//BATCH_SIZE,
    validation_data=(X_val_img, y_val_cat), 
    epochs=NUM_EPOCHS,
    callbacks=[early_stop]
)

# Slike + metapodaci

input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
input_meta = Input(shape=(train_meta.shape[1],)) 

# CNN Grana
x_img = Conv2D(32, (3,3), activation='relu')(input_img)
x_img = MaxPooling2D((2,2))(x_img)
x_img = Conv2D(64, (3,3), activation='relu')(x_img)
x_img = MaxPooling2D((2,2))(x_img)
x_img = Conv2D(128, (3,3), activation='relu')(x_img)
x_img = MaxPooling2D((2,2))(x_img)
x_img = Conv2D(256, (3,3), activation='relu', padding='same')(x_img) 
x_img = MaxPooling2D((2,2))(x_img)
x_img = Flatten()(x_img)
x_img = Dense(256, activation='relu')(x_img)

# Metapodaci Grana
x_meta = Dense(64, activation='relu')(input_meta)

x = Concatenate()([x_img, x_meta])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) 
output = Dense(NUM_CLASSES, activation='softmax')(x)

multi_model = Model(inputs=[input_img, input_meta], outputs=output)
multi_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_multi = multi_model.fit(
    [X_train_img, train_meta], y_train_cat, 
    validation_data=([X_val_img, val_meta], y_val_cat),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=[early_stop]
)

# Evaluacija svega

# Evalvacija FAZE 1
y_pred_probs_cnn = cnn_model.predict(X_test_img)
y_pred_cnn = np.argmax(y_pred_probs_cnn, axis=1)

acc_cnn = accuracy_score(y_test_enc, y_pred_cnn)
f1_cnn = f1_score(y_test_enc, y_pred_cnn, average='weighted', zero_division=0)

# Evalvacija FAZE 2
y_pred_probs_multi = multi_model.predict([X_test_img, test_meta])
y_pred_multi = np.argmax(y_pred_probs_multi, axis=1)

acc_multi = accuracy_score(y_test_enc, y_pred_multi)
f1_multi = f1_score(y_test_enc, y_pred_multi, average='weighted', zero_division=0)

# Tabela sa rezultatima
results = pd.DataFrame({
    'Model': ['Samo CNN', 'CNN + Metapodaci'],
    'Accuracy': [acc_cnn, acc_multi],
    'F1-score (Weighted)': [f1_cnn, f1_multi]
})

print("\n-----------------------------------------------")
print("          ANALIZA REZULTATA                      ")
print("-------------------------------------------------")
print(results.to_markdown(index=False))
print("-------------------------------------------------")

# Confusion Matrix za bolji model
final_preds = y_pred_multi if acc_multi > acc_cnn else y_pred_cnn
model_name = "CNN + Metapodaci" if acc_multi > acc_cnn else "Samo CNN"

cm = confusion_matrix(y_test_enc, final_preds)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_) 
plt.title(f'Confusion Matrix (Svi stilovi) za model: {model_name}')
plt.ylabel('Prave Klase')
plt.xlabel('Predviđene Klase')
plt.show()