import numpy as np

from backend.config import video_store_path
from backend.models.audio.audio_predict import audio_prediction
from backend.models.image.image_predict import image_prediction
from backend.service.audio_service import extract_audio_features
from backend.service.fusion_service import windowed_fusion
from backend.service.image_service import extract_image_features
from backend.utils.video_util import allowed_file

# extract images and audio features from video
def extract_image_and_audio_features(video_path):
    try:
        if not video_path:
            raise FileNotFoundError('No file provided')

        if not allowed_file(video_path):
            raise ValueError('Invalid file format')

        # audio_features
        audio_features = extract_audio_features(video_path)

        if audio_features is None or len(audio_features) == 0:
            raise ValueError('Audio features extraction failed')

        # image_features
        image_features = extract_image_features(video_path)

        if image_features is None or len(image_features) == 0:
            raise ValueError('Image features extraction failed')

        return audio_features, image_features

    except (FileNotFoundError, ValueError) as err:
        print('error: ', err)
        raise
    except Exception as ex:
        print('Unexpected error while extracting video features : ', ex)
        raise

# handles multimodal inference by processing image and audio features and performing fusion prediction.
def process_video_emotion_pipeline():
    # extracts features as well as preprocessing
    audio_features, image_features = extract_image_and_audio_features(video_store_path)

    # predict both audio and image model
    audio_vectors = audio_prediction(audio_features)
    image_vectors = image_prediction(image_features)

    # apply fusion to get stress scores
    stress_vector = windowed_fusion(audio_vectors, image_vectors)

    # final stress score and mapping
    stress_score, stress_level = map_stress_level(stress_vector)

    print('\nStress score: ', stress_score)
    print('\nStress level: ', stress_level)

# calculates final stress score and maps it
def map_stress_level(stress_vector):
    stress_vector = np.array(stress_vector)

    # smooth noise
    stress_vector = np.clip(stress_vector, 0, 1)

    # aggregate
    stress_score = (0.7 * np.mean(stress_vector) + 0.3 * np.max(stress_vector))

    # map level
    if stress_score < 0.4:
        stress_level = 'LOW'
    elif stress_score < 0.7:
        stress_level =  'MEDIUM'
    else:
        stress_level = 'HIGH'

    return stress_score, stress_level