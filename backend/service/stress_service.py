import numpy as np

from backend.utils.emotions_util import get_emotions, get_EMOTION_VA_MAP

emotions = get_emotions()
EMOTION_VA_MAP = get_EMOTION_VA_MAP()

# map emotion vector to stress score based on every fused window
def emotion_vector_to_stress(emotion_vector):
    try:
        emotion_vector = np.array(emotion_vector)

        # weighted VA point
        valence = sum(
            emotion_vector[i] * EMOTION_VA_MAP[emotions[i]][0]
            for i in range(len(emotion_vector))
        )

        arousal = sum(
            emotion_vector[i] * EMOTION_VA_MAP[emotions[i]][1]
            for i in range(len(emotion_vector))
        )

        # normalize (-1, 1) -> (0, 1)
        valence_norm = (valence + 1) / 2
        arousal_norm = (arousal + 1) / 2

        # stress = high arousal + low valence
        stress_score = (arousal_norm + (1 - valence_norm)) / 2

        print(f"  VA point      : valence={float(valence):.4f} arousal={float(arousal):.4f}")
        print(f"  stress score  : {float(stress_score):.4f}")

        return float(stress_score)

    except Exception as ex:
        print('Unexpected error while converting emotion vector to stress: ', ex)
        raise
