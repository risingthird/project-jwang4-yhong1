import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import util


def create_model():
    model = tf.keras.models.Sequential([
                # Adds a densely-connected layer with 64 units to the model:
                tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
                # Add another:
                tf.keras.layers.Dense(64, activation='relu'),
                # Add a softmax layer with 10 output units:
                tf.keras.layers.Dense(2, activation='softmax')])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    opts = util.parse_args()
    X, y = util.data_load(opts.dataset)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = create_model()
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        model.evaluate(X_test, y_test)


if __name__ == '__main__':
    main()

