import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import time

class EbayListingAnalyzer:
    def __init__(self, data_dir="ebay_data", models_dir="models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.embedding_model = None
        self.anomaly_detector = None
        self.price_predictor = None
        self.label_encoders = {}
        self.price_scaler = None
        self._setup_directories()
        
    def _setup_directories(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, 'encoders'), exist_ok=True)
        
    def load_data(self, master_csv=None):
        if not master_csv:
            csv_files = [f for f in os.listdir(self.data_dir) 
                        if f.startswith('ebay_listings_master_') and f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No master CSV files found")
            master_csv = os.path.join(self.data_dir, max(csv_files))
        return pd.read_csv(master_csv)
        
    def preprocess_data(self, df):
        # Handle missing values
        df['full_description'] = df['full_description'].fillna('')
        df['condition'] = df['condition'].fillna('Not Specified')
        df['shipping'] = df['shipping'].fillna(0)
        
        # Drop rows without prices
        df = df.dropna(subset=['price'])
        
        # Save original price for later reference
        df['original_price'] = df['price']
        
        # Encode categorical variables
        categorical_cols = ['condition', 'listing_type', 'query']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Scale prices using MinMaxScaler to keep them positive
        self.price_scaler = MinMaxScaler()
        df['price'] = self.price_scaler.fit_transform(df[['price']])
        
        return df
        
    def generate_embeddings(self, df, text_cols=['title', 'full_description'], 
                          model_name='all-MiniLM-L6-v2'):
        if not self.embedding_model:
            self.embedding_model = SentenceTransformer(model_name)
            
        embeddings = {}
        for col in text_cols:
            if col in df.columns:
                texts = df[col].fillna('').tolist()
                embeddings[col] = self.embedding_model.encode(texts, 
                                                           show_progress_bar=True,
                                                           batch_size=32)
                
        combined = np.hstack([emb for emb in embeddings.values()])
        pca = PCA(n_components=50)
        reduced_embeddings = pca.fit_transform(combined)
        
        # Add embedding features to dataframe
        for i in range(reduced_embeddings.shape[1]):
            df[f'embedding_{i}'] = reduced_embeddings[:, i]
            
        return df
        
    def detect_anomalies(self, df, contamination=0.1):
        feature_cols = ([col for col in df.columns if col.startswith('embedding_')] + 
                       ['price', 'shipping'])
        
        self.anomaly_detector = IsolationForest(contamination=contamination, 
                                              random_state=42)
        anomaly_labels = self.anomaly_detector.fit_predict(df[feature_cols])
        
        df['anomaly_score'] = self.anomaly_detector.score_samples(df[feature_cols])
        df['is_anomaly'] = anomaly_labels == -1
        
        return df
        
    def train_price_predictor(self, df, test_size=0.2):
        feature_cols = ([col for col in df.columns if col.startswith('embedding_')] + 
                       ['shipping'] + 
                       [col for col in df.columns if col.endswith('_encoded')])
        
        X = df[feature_cols]
        y = df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                           test_size=test_size, 
                                                           random_state=42)
        
        self.price_predictor = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.price_predictor.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        y_pred = self.price_predictor.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Price Prediction - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return {
            'rmse': rmse,
            'r2': r2,
            'feature_importance': self.price_predictor.feature_importances_,
            'feature_names': feature_cols
        }
    
    def save_models(self):
        joblib.dump(self.anomaly_detector, 
                   os.path.join(self.models_dir, 'anomaly_detector.joblib'))
        joblib.dump(self.price_predictor, 
                   os.path.join(self.models_dir, 'price_predictor.joblib'))
        joblib.dump(self.price_scaler, 
                   os.path.join(self.models_dir, 'price_scaler.joblib'))
        
        for name, encoder in self.label_encoders.items():
            joblib.dump(encoder, 
                       os.path.join(self.models_dir, 'encoders', f'{name}_encoder.joblib'))
            
    def analyze_new_listing(self, listing_data):
        df = pd.DataFrame([listing_data])
        df = self.preprocess_data(df)
        df = self.generate_embeddings(df)
        
        feature_cols = ([col for col in df.columns if col.startswith('embedding_')] + 
                       ['price', 'shipping'])
        anomaly_score = self.anomaly_detector.score_samples(df[feature_cols])[0]
        is_anomaly = self.anomaly_detector.predict(df[feature_cols])[0] == -1
        
        feature_cols = ([col for col in df.columns if col.startswith('embedding_')] + 
                       ['shipping'] + 
                       [col for col in df.columns if col.endswith('_encoded')])
        predicted_price = self.price_predictor.predict(df[feature_cols])[0]
        
        # Convert scaled price back to original scale
        predicted_price = self.price_scaler.inverse_transform([[predicted_price]])[0][0]
        
        return {
            'predicted_price': predicted_price,
            'anomaly_score': anomaly_score,
            'is_anomaly': is_anomaly
        }

def main():
    analyzer = EbayListingAnalyzer()
    
    try:
        # Load and process data
        df = analyzer.load_data()
        df = analyzer.preprocess_data(df)
        df = analyzer.generate_embeddings(df)
        df = analyzer.detect_anomalies(df)
        analyzer.train_price_predictor(df)
        analyzer.save_models()
        
        # Create simplified output
        results_df = pd.DataFrame({
            'title': df['title'],
            'url': df['url'],
            'original_price': df['original_price'],
            'anomaly_score': df['anomaly_score'],
            'is_anomaly': df['is_anomaly']
        })
        
        # Sort by anomaly score to highlight most unusual listings
        results_df = results_df.sort_values('anomaly_score')
        
        # Save minimal results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(analyzer.models_dir, f'anomaly_results_{timestamp}.csv')
        results_df.to_csv(results_file, index=False)
        
        # Print summary
        n_anomalies = results_df['is_anomaly'].sum()
        print(f"\nAnalysis Summary:")
        print(f"Total listings analyzed: {len(results_df)}")
        print(f"Anomalies detected: {n_anomalies} ({n_anomalies/len(results_df)*100:.1f}%)")
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nRuntime: {time.time() - start_time:.2f} seconds")