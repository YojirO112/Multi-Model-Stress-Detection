import joblib
import numpy as np

from backend.config import image_emotion_class_path, image_model_path

# image prediction
def image_prediction(image_features):
    try:
        print('Started image prediction ....')

        # loading emotion class and model
        emotion_class = joblib.load(image_emotion_class_path)
        # todo : load model
        model = joblib.load(image_model_path)

        predictions = model.predict(image_features)

        pred_idx = np.argmax(predictions, axis=1)
        emotions = [emotion_class[pred_idx[i]] for i in pred_idx]
        confidences = np.max(predictions, axis=1)

        print('image prediction results \n\n')

        for i in range(len(emotions)):
            print('segment : ', i + 1)
            print('prediction vector : ', predictions[i])
            print('emotion : ', emotions[i])
            print('confidence : ', round(confidences[i] * 100, 2), ' % \n\n')


    except FileNotFoundError as err:
        print('error : ', err)
        raise

    except Exception as ex:
        print('Unexpected Error during image inference : ', ex)
        raise
