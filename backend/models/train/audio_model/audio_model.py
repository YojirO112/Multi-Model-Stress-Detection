import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.losses import CategoricalCrossentropy

from backend.config import audio_dir, audio_model_path
from backend.models.evaluate_model import evaluate_model
from backend.models.preprocess.train.audio_preprocess.audio_generator import AudioDataGenerator
from backend.models.preprocess.train.audio_preprocess.audio_train_preprocess import load_dataset
from backend.models.train.audio_model.audio_Attention_Layer import AttentionLayer

# Architecture (EfficientNetB0 + Attention Layer + Dense)
def build_model(input_shape, num_classes):
    l2_reg = 0.0001

    #  Pretrained Backbone
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    x = base_model.output

    #  Attention
    x = AttentionLayer()(x)

    #  Dense Head
    x = layers.Dense(128,
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64,
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes,
                           activation='softmax',
                           kernel_initializer='glorot_uniform')(x)

    model = models.Model(base_model.input, outputs)

    return model, base_model

def train_model():
    try:
        print('Starting Training Audio Model ....')

        #  Load Data
        file_paths, y_onehot, y_encoded = load_dataset(audio_dir)

        X_train_paths, X_test_paths, y_train, y_test = train_test_split(
            file_paths, y_onehot,
            test_size=0.2,
            stratify=y_encoded,
            random_state=42
        )

        #  Generators
        train_gen = AudioDataGenerator(X_train_paths, y_train)
        test_gen = AudioDataGenerator(X_test_paths, y_test)

        #  Class Weights
        y_labels = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_labels),
            y=y_labels
        )
        class_weights = dict(enumerate(class_weights))

        # input
        input_shape = (128, 128, 3)
        num_classes = y_train.shape[1]

        model, base_model = build_model(input_shape, num_classes)

        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )

        print('Phase 1: Training Head (Frozen CNN)')

        # PHASE 1
        model.compile(
            optimizer = 'adam',
            loss = CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

        model.fit(
            train_gen,
            validation_data=test_gen,
            epochs = 40,
            class_weight=class_weights,
            callbacks=[early_stop, lr_scheduler]
        )

        print('Phase 2: Fine-Tuning CNN')

        # UNFREEZE TOP LAYERS
        for layer in base_model.layers[-40:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        model.compile(
            optimizer = 'adam',
            loss = CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

        history = model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=80,
            class_weight=class_weights,
            callbacks=[early_stop, lr_scheduler]
        )

        # Evaluate
        evaluate_model(history, model, test_gen, np.argmax(y_test, axis=1))

        model.save(audio_model_path)  # save model

        print('Audio model saved successfully')
        print("Audio model training completed ....")

    except Exception as ex:
        print('Unexpected error while training audio model:', ex)
        raise