import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.binary import BinaryEncoder


class BinaryEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.encoder = BinaryEncoder(cols=cols)

    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

    def get_feature_names_out(self, input_features=None):
        if hasattr(self.encoder, "get_feature_names"):
            return np.array(self.encoder.get_feature_names())
        return np.array([f"{col}_bin_{i}" for col in self.cols for i in range(10)])
