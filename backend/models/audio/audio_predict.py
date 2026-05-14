import joblib
import numpy as np
from tensorflow.python.keras.models import load_model

from backend.config import audio_encoder_path, audio_model_path
from backend.models.train.audio_model.audio_Attention_Layer import AttentionLayer

# audio prediction
def audio_prediction(audio_features):
    try:
        print('Started audio prediction ....')

        # load saved encoder and model
        encoder = joblib.load(audio_encoder_path)
        model = load_model(audio_model_path, custom_objects = {'AttentionLayer': AttentionLayer})

        predictions = model.predict(audio_features)

        pred_idx = np.argmax(predictions, axis = 1)
        emotions = encoder.inverse_transform(pred_idx)
        confidences = np.max(predictions, axis = 1)

        print('results \n\n')

        for i in range(len(emotions)):
            print('segment : ', i+1)
            print('prediction vector : ', predictions[i])
            print('emotion : ', emotions[i])
            print('confidence : ', round(confidences[i] * 100, 2), ' % \n\n')


    except FileNotFoundError as err:
        print('error : ', err)
        raise
    except Exception as ex:
        print('Unexpected Error during audio inference : ', ex)
        raise