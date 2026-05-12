import librosa
import numpy as np

from backend.models.preprocess.inference.audio_inference_preprocess import audio_inference_preprocess
from backend.models.preprocess.train.audio_preprocess.audio_generator import AudioDataGenerator
from backend.utils.audio_util import extract_audio, split_audio

# extract log-mel, delta, delta2 per segment
def extract_audio_features(video_path):
    try:
        audio_path = extract_audio(video_path)
        segments, sr = split_audio(audio_path)

        print('extracting audio features ....')

        if not segments:
            raise ValueError('No audio segments found')

        audio_features = []

        for segment in segments:
            feature_vector = AudioDataGenerator().process_audio(segment, sr)
            audio_features.append(feature_vector)

        audio_features = np.array(audio_features)
        print('audio features shape: ', audio_features.shape)

        # preprocess audio features for model inference
        audio_features = audio_inference_preprocess(audio_features)

        print('Audio features extracted successfully ....')

        return audio_features

    except ValueError as err:
        print('error : ', err)
        raise

    except Exception as ex:
        print('Unexpected Error while extracting audio features : ', ex)
        raise
