import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.feature_selection import RFECV

class HitSongPredictor:

    def __init__(self, features=None, target='popularity'):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.features = features or [
            'tempo', 'loudness', 'valence', 'energy',
            'danceability', 'acousticness', 'instrumentalness', 
            'liveness', 'speechiness', 'year', 'duration_ms', 
            'key', 'mode'  
        ]
        self.categorical_features = [] 
        self.target = target
        self.max_popularity = 100  
        self.selected_features = None 
        self.hit_threshold = 65  
        
    def preprocess_data(self, df):
        data = df.copy()

        for feature in self.features:
            if feature in data.columns and pd.api.types.is_numeric_dtype(data[feature]):
                data[feature] = data[feature].fillna(data[feature].median())

        y = None
        if self.target in data.columns:
            y = data[self.target].values
            if y is not None:
                self.max_popularity = max(np.max(y), self.max_popularity)

        if hasattr(self, 'selected_features') and self.selected_features is not None:
            available_features = [f for f in self.selected_features if f in data.columns]
            X = data[available_features].values
            return X, y, available_features
        
        available_features = [f for f in self.features if f in data.columns]
        X = data[available_features].values
        
        return X, y, available_features
        
    def train(self, df, hyperparameter_tuning=True, use_feature_selection=True, hit_threshold=65):

        self.hit_threshold = hit_threshold
        
        print("Preprocessing training data...")
        X, y, available_features = self.preprocess_data(df)
        
        self.available_features = available_features
        print(f"Training with features: {available_features}")
        
        self.max_popularity = max(np.max(y), self.max_popularity)
        print(f"Maximum popularity value: {self.max_popularity}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        self.scaler = StandardScaler().fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if use_feature_selection:
            print("Performing Recursive Feature Elimination...")
            base_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rfecv = RFECV(
                estimator=base_model,
                step=1,
                cv=5,
                scoring='neg_mean_squared_error',
                min_features_to_select=5,
                n_jobs=-1,
                verbose=1
            )
            rfecv.fit(X_train_scaled, y_train)
            
            selected_indices = rfecv.support_
            self.selected_features = [available_features[i] for i in range(len(available_features)) if selected_indices[i]]
            
            print(f"Selected {len(self.selected_features)}/{len(available_features)} features: {self.selected_features}")
            
            X_train_scaled = X_train_scaled[:, selected_indices]
            X_test_scaled = X_test_scaled[:, selected_indices]
        else:
            self.selected_features = available_features
        
        if hyperparameter_tuning:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [200, 400, 600],
                'max_depth': [None, 15, 30, 45],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42, oob_score=True),
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            print("Training model with default parameters...")
            self.model = RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features=None,
                random_state=42,
                n_jobs=-1,
                oob_score=True
            )
            self.model.fit(X_train_scaled, y_train)
            print(f"OOB Score: {self.model.oob_score_:.4f}")
        
        print("Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        if use_feature_selection:

            self.feature_importance = dict(zip(
                self.selected_features,
                self.model.feature_importances_
            ))
        else:
            self.feature_importance = dict(zip(
                available_features,
                self.model.feature_importances_
            ))
        
        self.feature_importance = {k: v for k, v in sorted(
            self.feature_importance.items(), 
            key=lambda item: item[1], 
            reverse=True
        )}
        
        print("Top 5 important features:")
        for feature, importance in list(self.feature_importance.items())[:5]:
            print(f"{feature}: {importance:.4f}")
        
        print("Computing classification metrics...")
        
        y_test_binary = (y_test >= hit_threshold).astype(int)
        y_pred_binary = (y_pred >= hit_threshold).astype(int)
        
        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        precision = precision_score(y_test_binary, y_pred_binary)
        recall = recall_score(y_test_binary, y_pred_binary)
        f1 = f1_score(y_test_binary, y_pred_binary)
        
        conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)
        
        class_report = classification_report(y_test_binary, y_pred_binary, target_names=['Non-Hit', 'Hit'])
        
        print(f"Classification metrics (threshold={hit_threshold}):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        print("\nClassification Report:")
        print(class_report)
        
        return {
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
            "feature_importance": self.feature_importance,
            "max_popularity": self.max_popularity,
            "selected_features": self.selected_features,
            "classification_metrics": {
                "threshold": hit_threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": class_report
            }
        }

    def predict(self, song_data):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(song_data, dict):
            song_data = pd.DataFrame([song_data])
        elif isinstance(song_data, pd.Series):
            song_data = pd.DataFrame([song_data])

        X, _, _ = self.preprocess_data(song_data)

        if hasattr(self, 'selected_features') and self.selected_features is not None:

            X_filtered = song_data[self.selected_features].values
            X = X_filtered

        X_scaled = self.scaler.transform(X)

        predicted_popularity = self.model.predict(X_scaled)[0]
        
        result = {
            "score": predicted_popularity,
            "isHit": predicted_popularity >= self.hit_threshold,
            "confidence": min(100, max(0, predicted_popularity)) / 100  # Add confidence score
        }
        
        explanation = self._generate_explanation_factors(song_data)
        result.update(explanation)
        
        return result
    
    def _get_value(self, df, column, default):

        if column not in df.columns:
            return default
            
        value = df[column].iloc[0]
        if isinstance(value, list):
            return value[0] if value else default
        return value
    
    def _generate_explanation_factors(self, song_data):
        
        positiveFactors = []
        negativeFactors = []
        
        top_features = list(self.feature_importance.keys())[:len(self.feature_importance)//2]
        
        for feature in top_features:

            if feature not in song_data.columns:
                continue
                
            value = song_data[feature].iloc[0]
            
            if feature == 'tempo':
                if 100 <= value <= 130:
                    positiveFactors.append(f"Tempo ({value:.1f} BPM) is in the optimal range for hit songs")
                elif value < 90 or value > 140:
                    negativeFactors.append(f"Tempo ({value:.1f} BPM) is outside typical hit range (100-130)")
            
            elif feature == 'loudness':
                if -8 <= value <= -2:
                    positiveFactors.append(f"Loudness ({value:.1f} dB) matches commercial standards")
                elif value < -12:
                    negativeFactors.append(f"Track may be too quiet ({value:.1f} dB) compared to hits")
            
            elif feature == 'valence':
                if value >= 0.5:
                    positiveFactors.append(f"Positive musical mood (valence: {value:.2f}) tends to perform well")
                elif value < 0.3:
                    negativeFactors.append(f"Low valence ({value:.2f}) may reduce mainstream appeal")
            
            elif feature == 'danceability':
                if value >= 0.6:
                    positiveFactors.append(f"High danceability ({value:.2f}) makes it more radio-friendly")
                elif value < 0.4:
                    negativeFactors.append(f"Low danceability ({value:.2f}) may reduce mainstream appeal")
            
            elif feature == 'energy':
                if value >= 0.6:
                    positiveFactors.append(f"Strong energy level ({value:.2f}) appeals to mainstream audiences")
                elif value < 0.4:
                    negativeFactors.append(f"Low energy level ({value:.2f}) may need boosting")
            
            elif feature == 'year':
                if value >= 2010:
                    positiveFactors.append(f"Recent release year ({value}) aligns with current trends")
                elif value < 2000:
                    negativeFactors.append(f"Older release year ({value}) may affect current popularity")
            
            elif feature == 'acousticness':
                if value <= 0.5 and value >= 0.1:
                    positiveFactors.append(f"Balanced acousticness ({value:.2f}) works well for mainstream")
                elif value > 0.8:
                    negativeFactors.append(f"High acousticness ({value:.2f}) may limit appeal in some markets")
            
            elif feature == 'instrumentalness':
                if value <= 0.05:
                    positiveFactors.append(f"Vocal-focused track (low instrumentalness: {value:.2f}) aligns with pop trends")
                elif value > 0.5:
                    negativeFactors.append(f"High instrumentalness ({value:.2f}) may limit vocal-focused audience")
        
        positiveFactors = positiveFactors[:5]
        negativeFactors = negativeFactors[:5]
        
        if not positiveFactors:
            positiveFactors.append("Track has some commercial potential elements")
        if not negativeFactors:
            negativeFactors.append("Consider analyzing similar hit songs in your target market")
        
        return {
            "positiveFactors": positiveFactors,
            "negativeFactors": negativeFactors
        }
    
    def evaluate_thresholds(self, X, y, threshold_range=range(50, 80, 5)):

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        y_pred = self.model.predict(X_scaled)
        
        results = {}
        for threshold in threshold_range:
            y_true_binary = (y >= threshold).astype(int)
            y_pred_binary = (y_pred >= threshold).astype(int)
            
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary)
            recall = recall_score(y_true_binary, y_pred_binary)
            f1 = f1_score(y_true_binary, y_pred_binary)
            
            results[threshold] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "hit_ratio": y_true_binary.mean() 
            }
        
        return results
    
    def plot_roc_curve(self, X, y, hit_threshold=None):

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if hit_threshold is None:
            hit_threshold = self.hit_threshold
        
        X_scaled = self.scaler.transform(X)
        
        y_pred = self.model.predict(X_scaled)
        
        y_true_binary = (y >= hit_threshold).astype(int)
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()
        
        return roc_auc
    
    def plot_classification_results(self, X, y, hit_threshold=None):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if hit_threshold is None:
            hit_threshold = self.hit_threshold
        
        X_scaled = self.scaler.transform(X)
        
        y_pred = self.model.predict(X_scaled)
        
        y_true_binary = (y >= hit_threshold).astype(int)
        y_pred_binary = (y_pred >= hit_threshold).astype(int)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = ['Non-Hit', 'Hit']
        tick_marks = [0, 1]
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([0, 100], [0, 100], '--', color='red')
        plt.plot([hit_threshold, hit_threshold], [0, 100], '--', color='green', label='Hit Threshold')
        plt.plot([0, 100], [hit_threshold, hit_threshold], '--', color='green')
        plt.xlabel('True Popularity')
        plt.ylabel('Predicted Popularity')
        plt.title('Predicted vs Actual Popularity')
        plt.legend()
        plt.savefig('prediction_scatter.png')
        plt.close()
    
    def save_model(self, filepath="hit_song_model.pkl"):
        if self.model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_importance": self.feature_importance,
            "features": self.features,
            "categorical_features": self.categorical_features,
            "max_popularity": self.max_popularity,
            "selected_features": self.selected_features,
            "hit_threshold": self.hit_threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="hit_song_model.pkl"):
        """Load a trained model from a file and restore scaler, features, and settings."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_importance = model_data["feature_importance"]
        self.features = model_data["features"]
        self.categorical_features = model_data["categorical_features"]
        self.max_popularity = model_data.get("max_popularity", 100)
        self.selected_features = model_data.get("selected_features")
        self.hit_threshold = model_data.get("hit_threshold", 65)

        print(f"Model loaded from {filepath}")
        print(f"Model trained with {len(self.selected_features or self.features)} features")
        if self.selected_features:
            print(f"Selected features: {self.selected_features}")