import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

img_size = (256, 256)

# image preprocessing for model inference
def image_preprocess_inference(image_paths):
    try:
        print('Preprocessing images for inference ....')

        processed = []

        for img_path in image_paths:
            # reading each image path
            img = cv2.imread(img_path)

            if img is None:
                raise ValueError(f'Could not read image {img_path}')

            # resizing = (256, 256)
            img = cv2.resize(img, img_size)

            # Converting BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # rescaling each image 0-255 -> (-1, 1)
            img = img.astype("float32")
            img = preprocess_input(img)

            processed.append(img)

        # stacking (num_images, 256, 256, 3)
        processed = np.array(processed)

        print('Completed Preprocessing images for inference ....')

        return processed

    except ValueError as err:
        print('Error: ', err)
        raise
    except Exception as ex:
        print('Unexpected Error while preprocessing images for inferencing', ex)
        raise