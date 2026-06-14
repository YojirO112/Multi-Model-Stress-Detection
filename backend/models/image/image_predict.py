import joblib
import numpy as np
from tensorflow.keras.models import load_model

from backend.config import image_emotion_class_path, image_model_path
from backend.models.train.image_model.CBAM_attention_layer import CBAM

# image prediction
def image_prediction(image_features):
    try:
        print('Started image prediction ....')

        # loading emotion class and model
        emotion_class = joblib.load(image_emotion_class_path)
        model = load_model(image_model_path,
                           custom_objects = {'CBAM': CBAM},
                           compile = False)

        predictions = model.predict(image_features)

        pred_idx = np.argmax(predictions, axis=1)
        # # emotions = [emotion_class[pred_idx[i]] for i in pred_idx]
        # emotions = [emotion_class[idx] for idx in pred_idx]
                # Invert the dictionary to map index -> emotion name
        emotion_labels = {v: k for k, v in emotion_class.items()}
        emotions = [emotion_labels[idx] for idx in pred_idx]

        confidences = np.max(predictions, axis=1)

        print('image prediction results \n\n')

        for i in range(len(emotions)):
            print('segment : ', i + 1)
            print('prediction vector : ', predictions[i])
            print('emotion : ', emotions[i])
            print('confidence : ', round(confidences[i] * 100, 2), ' % \n\n')

        return predictions

    except FileNotFoundError as err:
        print('error : ', err)
        raise

    except Exception as ex:
        print('Unexpected Error during image inference : ', ex)
        raise
