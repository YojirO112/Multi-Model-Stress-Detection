from backend.config import video_store_dir
from backend.models.audio.audio_predict import audio_prediction
from backend.models.image.image_predict import image_prediction
from backend.service.audio_service import extract_audio_features
from backend.service.fusion_service import windowed_fusion
from backend.service.image_service import extract_image_features
from backend.utils.video_util import allowed_file

# extract video features (images + audio)
def extract_video_features(video_path):
    try:
        if not video_path:
            raise FileNotFoundError('No file provided')

        if not allowed_file(video_path):
            raise ValueError('Invalid file format')

        # audio_features
        audio_features = extract_audio_features(video_path)

        if not audio_features:
            raise ValueError('Audio features extraction failed')

        # image_features
        image_features = extract_image_features(video_path)

        if not image_features:
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
    audio_features, image_features = extract_video_features(video_store_dir)

    # predict both audio and image model
    audio_vectors = audio_prediction(audio_features)
    image_vectors = image_prediction(image_features)

    # apply fusion to get stress scores
    stress_scores = windowed_fusion(audio_vectors, image_vectors)

    # todo : strategy to map scores to stress levels