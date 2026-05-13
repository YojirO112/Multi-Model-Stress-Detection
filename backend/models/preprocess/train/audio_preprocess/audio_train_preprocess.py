import os

import joblib
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from backend.config import audio_encoder_path


# loading audio datasets from audio dir
def load_dataset(dataset_path):
    print('Loading Audio Dataset ....')

    file_paths = []
    labels = []

    for emotion in os.listdir(dataset_path):
        emotion_path = os.path.join(dataset_path, emotion)

        if not os.path.isdir(emotion_path):
            continue

        for file in os.listdir(emotion_path):
            file_paths.append(os.path.join(emotion_path, file))
            labels.append(emotion)

    # encoding labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels)
    y_onehot = to_categorical(y_encoded)

    # saving encoder
    joblib.dump(encoder, audio_encoder_path)

    print('Finished Loading Audio Dataset ....')

    return file_paths, y_onehot, y_encoded