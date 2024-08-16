# %% Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# %% Load data
df = pd.read_pickle("../data/processed/2_second_processed_merged_df.pkl")

# %% Drop NaN values and duplicates
df = df.dropna().drop_duplicates(subset=['Full_seq_dna_parent', 'Full_seq_dna_child', 'target'], keep='first').reset_index(drop=True)

# %% Encode the sequences
tokenizer = Tokenizer(char_level=True)  # Character-level tokenizer
tokenizer.fit_on_texts(df['Full_seq_dna_parent'] + df['Full_seq_dna_child'])

# Integer encode the sequences
X_parent = tokenizer.texts_to_sequences(df['Full_seq_dna_parent'])
X_child = tokenizer.texts_to_sequences(df['Full_seq_dna_child'])

# Pad the sequences
max_seq_length = 2000  # Maximum length for each DNA sequence
X_parent = pad_sequences(X_parent, maxlen=max_seq_length, padding='post')
X_child = pad_sequences(X_child, maxlen=max_seq_length, padding='post')

# Combine parent and child sequences
X = np.concatenate((X_parent, X_child), axis=1)

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(df['target'])

# %% Apply SMOTE to balance the dataset
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=50, n_jobs=-1)
X, y = smote.fit_resample(X, y)

# %% Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# %% Define the LSTM model
embedding_dim = 128
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_seq_length * 2),
    Bidirectional(LSTM(64, return_sequences=True)),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# %% Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %% Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# %% Evaluate the model
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# %% Print evaluation metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
