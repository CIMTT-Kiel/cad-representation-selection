import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin

class LogScaler(BaseEstimator, TransformerMixin):
    """
    Log-Scaler
    Transformiert Daten mit np.log() und kann rücktransformieren.
    """
    
    def __init__(self, epsilon=1e-8):
        """
        Parameters:
        -----------
        epsilon : float, default=1e-8
            Kleiner Wert der zu den Daten addiert wird um log(0) auszuschlißen
        """
        self.epsilon = epsilon
        self.fitted_ = False
        
    def fit(self, X, y=None):
        """
        placeholder fit Methode - log scaler benötigt kein fitting aber spart code in der pipeline bei der Auswahl
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        """
        X = self._validate_data(X)
        
        # Prüfe auf negative oder null Werte
        if np.any(X <= 0):
            print(f"Warning: Found {np.sum(X <= 0)} values <= 0. Adding epsilon={self.epsilon}")
        
        self.fitted_ = True
        return self
        
    def transform(self, X):
        """
        Transform mit Log
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : ndarray of shape (n_samples, n_features)
            Log-transformed data
        """
        if not self.fitted_:
            raise ValueError("This LogScaler instance is not fitted yet.")
            
        X = self._validate_data(X)
        
        # Log-Transformation mit epsilon für Sicherheit
        X_transformed = np.log(X + self.epsilon)
        
        return X_transformed
        
    def inverse_transform(self, X):
        """
        Rück-Transformation (exp)
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Log-transformed data
            
        Returns:
        --------
        X_original : ndarray of shape (n_samples, n_features)
            Original scale data
        """
        if not self.fitted_:
            raise ValueError("This LogScaler instance is not fitted yet.")
            
        X = self._validate_data(X)
        
        # Rück-transformation
        X_original = np.exp(X) - self.epsilon
        
        return X_original
        
    def _validate_data(self, X):
        """Helper function für Datenvalidierung"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X
