import numpy as np
from scipy.stats import entropy

from backend.service.emotion_service import emotion_vector_to_stress
from backend.utils.emotions_util import get_emotions

# calculate entropy of model to know its certainty
def model_certainty(vector):
     e = entropy(vector)
     max_entropy = np.log(len(vector))

     return float(1 - (e / max_entropy))

# single window fusion for 1 audio vector and 1 averaged image vector
def fuse_window(audio_vector, image_vector):
    try:
        audio_vector = np.array(audio_vector)
        image_vector = np.array(image_vector)

        # confidence based wights
        audio_certainty = model_certainty(audio_vector)
        image_certainty = model_certainty(image_vector)

        total_certainty = audio_certainty + image_certainty
        audio_weight = audio_certainty / total_certainty
        image_weight = image_certainty / total_certainty

        # strategy - 1 (confidence weighted average)
        conf_weight = (audio_weight * audio_vector) + (image_weight * image_vector)
        conf_weight /= conf_weight.sum() # normalize

        # strategy - 2 (max fusion)
        max_fusion = np.maximum(audio_vector, image_vector)
        max_fusion /= max_fusion.sum()

        # strategy - 3 (agreement fusion)
        audio_idx = np.argmax(audio_vector)
        image_idx = np.argmax(image_vector)

        # agreement on same emotion
        if audio_idx == image_idx:
            boost = 1 + (total_certainty / 2) # boost by mean certainty
            agreement_fusion = conf_weight.copy()

            agreement_fusion[audio_idx] *= boost
            agreement_fusion /= np.sum(agreement_fusion)
            print(f'models agree on: {get_emotions()[audio_idx]}, boost: {boost:.4f}')

            # combine all 3 strategies by their confidence
            weighted_conf = model_certainty(conf_weight)
            max_conf = model_certainty(max_fusion)
            agreement_conf = model_certainty(agreement_fusion)

            total_conf = weighted_conf + max_conf + agreement_conf
            w1 = weighted_conf / total_conf
            w2 = max_conf / total_conf
            w3 = agreement_conf / total_conf

            final_vector = (w1 * conf_weight) + (w2 * max_fusion) + (w3 * agreement_fusion)

        else: # disagreement
            agreement_fusion = np.zeros(len(get_emotions()))
            print(f'models disagree -> audio: {get_emotions()[audio_idx]}, image: {get_emotions()[image_idx]}')

            # combine all 2 strategies by their confidence
            weighted_conf = model_certainty(conf_weight)
            max_conf = model_certainty(max_fusion)

            total_conf = weighted_conf + max_conf
            w1 = weighted_conf / total_conf
            w2 = max_conf / total_conf

            final_vector = (w1 * conf_weight) + (w2 * max_fusion)

        final_vector /= final_vector.sum()

        print('Fusion for window: ', final_vector)

        return final_vector

    except Exception as ex:
        print('Unexpected error while fusing window: ', ex)
        raise

# aggregate image vectors in window
def aggregate_image_vectors(image_vectors):
    try:
        vectors = np.array(image_vectors)

        # confidence weighted average
        certainties = np.array([model_certainty(v) for v in vectors])

        # normalize
        weights = certainties/ np.sum(certainties)

        # weighted average across images
        aggregated = np.average(vectors, axis = 0, weights = weights)
        aggregated /= np.sum(aggregated)

        print(f'Aggregated {len(vectors)} frames -> {aggregated.round(4)}')

        return aggregated

    except Exception as ex:
        print('Unexpected error while aggregating image vectors: ', ex)
        raise

# align audio and image vectors based by time window
def windowed_fusion(audio_vectors, image_vectors, fps = 1, segment_duration = 2):
    try:
        print('Executing windowed fusion ....')

        audio_vectors = np.array(audio_vectors)
        image_vectors = np.array(image_vectors)

        len_audio = len(audio_vectors)
        len_image = len(image_vectors)

        # 1 fps * 2 segment duration = 2 images per window
        images_per_window = fps * segment_duration

        stress_vectors = []
        emotion_vectors = []

        for window in range(len_audio):
            audio_vector = audio_vectors[window]

            # get image vectors for this window
            image_start = window * images_per_window
            image_end = image_start + images_per_window

            window_images = image_vectors[image_start: image_end] # (3, 7)

            # checks for 0 image vectors and skips it
            if len(window_images) == 0:
                print(f'No images found for window {window}, skipping')
                continue

            # skip aggregation if image vector length = 1
            if len(window_images) == 1:
                image_vector = window_images[0]
            else: # average frames
                image_vector = aggregate_image_vectors(window_images)

            # fuse window based on audio and image vector
            fused_emotion = fuse_window(audio_vector, image_vector)

            # calculate stress from emotion fused vector
            stress_score = emotion_vector_to_stress(fused_emotion)

            stress_vectors.append(stress_score)
            emotion_vectors.append(fused_emotion)

        stress_vector = np.array(stress_vectors)  # (N_windows,)
        emotion_vectors = np.array(emotion_vectors)  # (N_windows, 7)

        print(f"\nStress vector   : {stress_vector.round(4)}")
        print(f"Emotion vectors : {emotion_vectors.shape}")

        return stress_vector

    except Exception as ex:
        print('Unexpected error while executing windowed_fusion: ', ex)
        raise