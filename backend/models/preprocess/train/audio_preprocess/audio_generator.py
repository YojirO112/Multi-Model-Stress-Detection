import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.utils import Sequence

# convert audio to (log-mel, delta, delta²)
class AudioDataGenerator(Sequence):
    def __init__(self, file_paths = None, labels = None,
                 batch_size=32,
                 sr=22050,
                 n_mels=128,
                 max_len=128,
                 shuffle=True):

        self.file_paths = file_paths or []
        self.labels = labels
        self.batch_size = batch_size
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_paths = [self.file_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]

        X = self.__data_generation(batch_paths)

        return np.array(X), np.array(batch_labels)

    # Core function
    def __data_generation(self, batch_paths):
        X = []

        for file_path in batch_paths:
            y, sr = librosa.load(file_path, sr=self.sr) # Loads waveform
            spec = self.process_audio(y, sr)
            X.append(spec)

        return X

    # FULL AUDIO PIPELINE
    def process_audio(self, y, sr):
        #  Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        #  Delta
        delta = librosa.feature.delta(mel_db)

        #  Delta²
        delta2 = librosa.feature.delta(mel_db, order=2)

        # Normalize
        mel_db = self.normalize(mel_db)
        delta = self.normalize(delta)
        delta2 = self.normalize(delta2)

        #  Stack (3-channel)
        spec = np.stack([mel_db, delta, delta2], axis=-1)

        #  Pad / Trim
        spec = self.pad_or_trim(spec)

        return spec.astype(np.float32)

    #  Normalization
    def normalize(self, x):
        return (x - np.mean(x)) / (np.std(x) + 1e-6)

    #  Fix size to (128,128,3)
    def pad_or_trim(self, spec):
        if spec.shape[1] < self.max_len:
            pad_width = self.max_len - spec.shape[1]
            spec = np.pad(spec, ((0,0),(0,pad_width),(0,0)), mode='constant')
        else:
            spec = spec[:, :self.max_len, :]

        return spec