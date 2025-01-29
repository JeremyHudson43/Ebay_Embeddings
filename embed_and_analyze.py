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
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class EbayListingAnalyzer:
    def __init__(self, data_dir="ebay_data", models_dir="models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.embedding_model = None
        self.image_model = None
        self.anomaly_detector = None
        self.price_predictor = None
        self.label_encoders = {}
        self.price_scaler = None
        self._setup_directories()
        self._setup_image_processor()
        
    def _setup_directories(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, 'encoders'), exist_ok=True)
    
    def _setup_image_processor(self):
        # Load pre-trained ResNet model
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.eval()  # Set to evaluation mode
        
        # Remove the final classification layer
        self.image_model = torch.nn.Sequential(*list(self.image_model.children())[:-1])
        
        if torch.cuda.is_available():
            self.image_model = self.image_model.cuda()
            
        # Define image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def load_data(self, master_csv=None):
        if not master_csv:
            csv_files = [f for f in os.listdir(self.data_dir) 
                        if f.startswith('ebay_listings_master_') and f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No master CSV files found")
            master_csv = os.path.join(self.data_dir, max(csv_files))
        return pd.read_csv(master_csv)
        
    def process_image(self, image_path):
        """Extract features from a single image using ResNet"""
        try:
            with Image.open(image_path).convert('RGB') as img:
                img_tensor = self.image_transforms(img)
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                
                with torch.no_grad():
                    features = self.image_model(img_tensor.unsqueeze(0))
                    return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
            
    def get_image_embeddings(self, df):
        """Generate embeddings for all images of each listing"""
        image_embeddings = []
        
        for _, row in df.iterrows():
            if pd.isna(row['image_folder']) or not row['image_folder']:
                # No images available
                image_embeddings.append(np.zeros(2048))  # ResNet feature size
                continue
                
            # Get all images for this listing
            img_dir = os.path.join(row['full_filepath'])
            if not os.path.exists(img_dir):
                image_embeddings.append(np.zeros(2048))
                continue
                
            # Process all images and average their embeddings
            valid_embeddings = []
            image_files = [f for f in os.listdir(img_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
            for img_file in image_files[:5]:  # Limit to first 5 images
                img_path = os.path.join(img_dir, img_file)
                features = self.process_image(img_path)
                if features is not None:
                    valid_embeddings.append(features)
            
            if valid_embeddings:
                # Average the embeddings from all images
                avg_embedding = np.mean(valid_embeddings, axis=0)
                image_embeddings.append(avg_embedding)
            else:
                image_embeddings.append(np.zeros(2048))
        
        return np.array(image_embeddings)
    
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
        print("Generating text embeddings...")
        if not self.embedding_model:
            self.embedding_model = SentenceTransformer(model_name)
            
        # Generate text embeddings
        embeddings = {}
        for col in text_cols:
            if col in df.columns:
                texts = df[col].fillna('').tolist()
                embeddings[col] = self.embedding_model.encode(texts, 
                                                           show_progress_bar=True,
                                                           batch_size=32)
                
        # Combine text embeddings
        text_embeddings = np.hstack([emb for emb in embeddings.values()])
        
        print("Generating image embeddings...")
        # Generate image embeddings
        image_embeddings = self.get_image_embeddings(df)
        
        # Combine text and image embeddings
        combined = np.hstack([text_embeddings, image_embeddings])
        
        # Reduce dimensionality
        print("Reducing embedding dimensions...")
        pca = PCA(n_components=100)  # Increase components due to additional features
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
        
        # Generate predictions for all data
        all_predictions = self.price_predictor.predict(X)
        # Convert predictions back to original scale
        df['predicted_price'] = self.price_scaler.inverse_transform(all_predictions.reshape(-1, 1))
        
        # Calculate metrics on test set
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
        predicted_price = self.price_scaler.inverse_transform([[predicted_price]])[0][0]
        
        return {
            'predicted_price': predicted_price,
            'anomaly_score': anomaly_score,
            'is_anomaly': is_anomaly
        }

def main():
    analyzer = EbayListingAnalyzer()
    
    try:
        print("Loading and processing data...")
        df = analyzer.load_data()
        df = analyzer.preprocess_data(df)
        
        print("Generating embeddings (this may take a while)...")
        df = analyzer.generate_embeddings(df)
        
        print("Detecting anomalies...")
        df = analyzer.detect_anomalies(df)
        
        print("Training price predictor...")
        analyzer.train_price_predictor(df)
        
        print("Saving models...")
        analyzer.save_models()
        
        # Create simplified output
        results_df = pd.DataFrame({
            'title': df['title'],
            'url': df['url'],
            'actual_price': df['original_price'],
            'predicted_price': df['predicted_price'],
            'price_difference': df['original_price'] - df['predicted_price'],
            'price_delta_pct': ((df['original_price'] - df['predicted_price']) / df['predicted_price'] * 100).round(2),
            'anomaly_score': df['anomaly_score'],
            'is_anomaly': df['is_anomaly']
        })
        
        # Create two sorted views
        anomaly_sorted = results_df.sort_values('anomaly_score')
        delta_sorted = results_df.sort_values('price_delta_pct', key=abs, ascending=False)
        
        # Save both views
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        anomaly_sorted.to_csv(os.path.join(analyzer.models_dir, f'anomaly_results_{timestamp}.csv'), index=False)
        delta_sorted.to_csv(os.path.join(analyzer.models_dir, f'price_delta_results_{timestamp}.csv'), index=False)
        
        # Print top price discrepancies
        print("\nTop 5 Largest Price Discrepancies:")
        for _, row in delta_sorted.head().iterrows():
            print(f"\nTitle: {row['title']}")
            print(f"Actual Price: ${row['actual_price']:.2f}")
            print(f"Predicted Price: ${row['predicted_price']:.2f}")
            print(f"Delta: {row['price_delta_pct']:+.1f}%")
        
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