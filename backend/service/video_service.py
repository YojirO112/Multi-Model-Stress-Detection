from backend.service.audio_service import extract_audio_features
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