import numpy as np
from keras.src.applications.efficientnet import EfficientNetB0

from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam

from backend.config import image_model_path
from backend.models.evaluate_model import evaluate_model
from backend.models.preprocess.train.data_augmentation import image_generator
from backend.models.train.image_model.CBAM_attention_layer import CBAM
from backend.utils.emotions_util import get_emotions


# Architecture (EfficientNetB0 + CBAM + Dense)
def build_image_model(input_shape, num_classes):
    l2_reg = 0.0001

    #  Backbone
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    x = base_model.output

    # CBAM attention layer
    x = CBAM()(x)

    #  Head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256,
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128,
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes,
                           activation='softmax',
                           kernel_initializer='glorot_uniform')(x)

    model = models.Model(base_model.input, outputs)

    return model, base_model

# train image model
def train_image_model():
    try:
        print("Building Image Model...")

        # Build Model
        model, base_model = build_image_model(input_shape=(256, 256, 3), num_classes=len(get_emotions()))
        model.summary()

        #  Generators
        train_gen, test_gen = image_generator()

        # Class Weights
        y_train = train_gen.classes
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(class_weights))

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=8,
                min_delta=1e-4,
                restore_best_weights=False,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=4,
                cooldown=2,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # PHASE 1 (Frozen CNN)
        print("Phase 1: Training Head (Frozen Backbone)")

        model.compile(
            optimizer = Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=20,
            class_weight=class_weights,
            callbacks=callbacks
        )

        # PHASE 2 (Fine-Tuning)
        print("Phase 2: Fine-Tuning Backbone")

        # Unfreeze top layers
        for layer in base_model.layers[-40:]:
            layer.trainable = True

        model.compile(
            optimizer = Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=30,
            class_weight=class_weights,
            callbacks=callbacks
        )

        # Evaluate
        evaluate_model(history)

        model.save(image_model_path) # save model

        print('Image model saved successfully')
        print("Image model training completed ....")

    except Exception as ex:
        print('Unexpected error while training image model:', ex)
        raise