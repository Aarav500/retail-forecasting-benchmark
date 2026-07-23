"""LSTM baseline — 2-layer LSTM over a MinMax-scaled lagged window.

Ported from `code/experiment.py:run_lstm`.
"""
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from .base import BaseForecaster, Forecast


class LSTMForecaster(BaseForecaster):
    def __init__(self, lags: int = 14, units_layer1: int = 64, units_layer2: int = 32,
                 dropout: float = 0.2, epochs: int = 100, batch_size: int = 16,
                 patience: int = 10, validation_split: float = 0.1, random_seed: int = 42):
        super().__init__()
        self.lags = lags
        self.units_layer1 = units_layer1
        self.units_layer2 = units_layer2
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.name = "LSTM"
        self._model = None
        self._scaler = None
        self._train_scaled: list[float] = []

    def fit(self, train: pd.Series) -> "LSTMForecaster":
        tf.random.set_seed(self.random_seed)
        t0 = time.time()
        values = train.values.astype(float)
        self._scaler = MinMaxScaler()
        train_scaled = self._scaler.fit_transform(values.reshape(-1, 1)).flatten()

        X, y = [], []
        for i in range(self.lags, len(train_scaled)):
            X.append(train_scaled[i - self.lags:i])
            y.append(train_scaled[i])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        self._model = Sequential([
            LSTM(self.units_layer1, return_sequences=True, input_shape=(self.lags, 1)),
            Dropout(self.dropout),
            LSTM(self.units_layer2),
            Dropout(self.dropout),
            Dense(1),
        ])
        self._model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(patience=self.patience, restore_best_weights=True, verbose=0)
        self._model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                         validation_split=self.validation_split, callbacks=[es], verbose=0)
        self._train_scaled = list(train_scaled)
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        t0 = time.time()
        history = list(self._train_scaled)
        test_values = test.values.astype(float)
        forecasts_scaled = []
        for i in range(len(test_values)):
            x = np.array(history[-self.lags:]).reshape(1, self.lags, 1)
            pred_scaled = self._model.predict(x, verbose=0)[0, 0]
            forecasts_scaled.append(pred_scaled)
            test_scaled = self._scaler.transform([[test_values[i]]])[0, 0]
            history.append(test_scaled)

        forecasts = self._scaler.inverse_transform(
            np.array(forecasts_scaled).reshape(-1, 1)
        ).flatten()
        self.pred_time_ = time.time() - t0
        return Forecast(point=forecasts)
