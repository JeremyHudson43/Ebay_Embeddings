import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from datetime import datetime

class DataPreparer:
    def __init__(self, base_dir="ebay_data"):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, "train")
        self.test_dir = os.path.join(base_dir, "test")
        
    def setup_directories(self):
        """Create train and test directories"""
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create images subdirectories
        os.makedirs(os.path.join(self.train_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "images"), exist_ok=True)

    def find_master_csv(self):
        """Find the most recent master CSV file"""
        csv_files = [f for f in os.listdir(self.base_dir) 
                    if f.startswith("ebay_listings_master_") and f.endswith(".csv")]
        
        if not csv_files:
            raise FileNotFoundError("No master CSV files found")
            
        # Sort by date in filename
        latest_file = sorted(csv_files)[-1]
        return os.path.join(self.base_dir, latest_file)

    def copy_image_folder(self, src_folder, dest_base, new_relative_path):
        """Copy image folder to new location and return new path"""
        if not src_folder or pd.isna(src_folder):
            return None
            
        src_path = os.path.join(self.base_dir, src_folder)
        if not os.path.exists(src_path):
            print(f"Warning: Source path does not exist: {src_path}")
            return None
            
        dest_path = os.path.join(dest_base, "images", new_relative_path)
        try:
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            return os.path.relpath(dest_path, dest_base)
        except Exception as e:
            print(f"Error copying {src_path}: {e}")
            return None

    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare train and test datasets"""
        print("Setting up directories...")
        self.setup_directories()
        
        print("Loading master CSV...")
        master_csv = self.find_master_csv()
        df = pd.read_csv(master_csv)
        
        # Clean the data
        print("Cleaning data...")
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
        
        # Optional: Remove outliers
        q1 = df['price'].quantile(0.01)
        q3 = df['price'].quantile(0.99)
        df = df[(df['price'] >= q1) & (df['price'] <= q3)]
        
        # Split the data
        print("Splitting into train and test sets...")
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            random_state=random_state,
            stratify=df['query'] if 'query' in df.columns else None
        )
        
        # Process train set
        print("Processing train set...")
        train_df['new_image_folder'] = train_df.apply(
            lambda row: self.copy_image_folder(
                row['image_folder'],
                self.train_dir,
                f"{row.name}_" + os.path.basename(str(row['image_folder'])) if pd.notna(row['image_folder']) else None
            ),
            axis=1
        )
        
        # Process test set
        print("Processing test set...")
        test_df['new_image_folder'] = test_df.apply(
            lambda row: self.copy_image_folder(
                row['image_folder'],
                self.test_dir,
                f"{row.name}_" + os.path.basename(str(row['image_folder'])) if pd.notna(row['image_folder']) else None
            ),
            axis=1
        )
        
        # Update image folder paths and save CSVs
        for df, name, path in [(train_df, 'train', self.train_dir), 
                             (test_df, 'test', self.test_dir)]:
            df['image_folder'] = df['new_image_folder']
            df = df.drop(columns=['new_image_folder'])
            
            output_file = os.path.join(path, f"{name}_set.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved {name} set to {output_file}")
        
        # Print summary statistics
        print("\nDataset Summary:")
        print(f"Total samples: {len(df)}")
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Calculate price statistics
        print("\nPrice Statistics:")
        stats = {
            'Training Set': train_df['price'].describe(),
            'Test Set': test_df['price'].describe()
        }
        print(pd.DataFrame(stats))
        
        return train_df, test_df

def main():
    preparer = DataPreparer()
    
    print("Starting data preparation...")
    train_df, test_df = preparer.prepare_data()
    
    # Print some sample rows
    print("\nSample from training set:")
    print(train_df[['title', 'price', 'image_folder']].head())
    
    print("\nSample from test set:")
    print(test_df[['title', 'price', 'image_folder']].head())

if __name__ == "__main__":
    main()