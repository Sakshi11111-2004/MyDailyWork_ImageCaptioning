import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Add, Flatten, RepeatVector
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# Model parameters
vocab_size = 7  # Number of unique words in your vocabulary
max_length = 6  # Maximum length of sequences
embedding_dim = 256
lstm_units = 256

# Define pre-trained VGG16 model for feature extraction
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg_model.layers:
    layer.trainable = False

# Define image feature extraction model
img_input = Input(shape=(224, 224, 3))
x = vgg_model(img_input)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)  # Project to match LSTM units
x = RepeatVector(max_length)(x)  # Repeat the vector to match sequence length

# Define caption input
caption_input = Input(shape=(max_length,))

# Define caption processing
embedding = Embedding(vocab_size, embedding_dim)(caption_input)
dropout = Dropout(0.5)(x)
lstm = LSTM(lstm_units, return_sequences=True)(embedding)
add = Add()([dropout, lstm])
output = Dense(vocab_size, activation='softmax')(add)

# Define the final model
model = Model(inputs=[img_input, caption_input], outputs=output)

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

# Model summary
model.summary()

def preprocess_image(image_path):
    """Load and preprocess an image."""
    image = load_img(image_path, target_size=(224, 224))  # Resize image to 224x224
    image = img_to_array(image)  # Convert image to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize image
    return image

def one_hot_encode_caption(caption, vocab_size, max_length):
    """One-hot encode a caption."""
    # Example: convert caption list of word indices to one-hot encoding
    caption_seq = np.zeros((max_length, vocab_size))
    for i, word_index in enumerate(caption):
        if i < max_length:
            caption_seq[i, word_index] = 1
    return caption_seq

# Example usage with an actual image and caption
image_path = r"C:\Users\SAKSHI\OneDrive\Pictures\image1.jpg"  # Replace with the path to your image
img_feature = preprocess_image(image_path)

# Example caption sequence (replace with actual caption processing)
caption_sequence = [0, 1, 2, 3, 4, 5]  # Replace with your actual caption indices
caption_sequence_padded = pad_sequences([caption_sequence], maxlen=max_length, padding='post')[0]
next_word = one_hot_encode_caption(caption_sequence_padded, vocab_size, max_length)

# Training step
model.train_on_batch([img_feature, np.expand_dims(caption_sequence_padded, axis=0)], np.expand_dims(next_word, axis=0))
print("Image feature shape:", img_feature.shape)
print("Caption sequence shape:", caption_sequence_padded.shape)
print("Next word shape:", next_word.shape)
