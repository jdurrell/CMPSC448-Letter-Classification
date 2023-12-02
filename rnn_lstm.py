import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from emnist import extract_training_samples, extract_test_samples

# Load EMNIST dataset
train_images, train_labels = extract_training_samples('letters')
test_images, test_labels = extract_test_samples('letters')

# Verify the number of classes in the dataset
num_classes = 47  # EMNIST "letters" has 47 classes

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], -1, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], -1, 1)).astype('float32') / 255

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

# Build a more complex RNN model
model = models.Sequential()
model.add(layers.LSTM(64, input_shape=(train_images.shape[1], 1), return_sequences=True))
model.add(layers.BatchNormalization())
model.add(layers.LSTM(32))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with more epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model.fit(train_images, train_labels, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

