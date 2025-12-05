# src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

class KenyanMarketConverter:
    def __init__(self):
        self.kenyan_factors = {
            'price_adjustment': 0.3,  # Reduce prices for Kenyan market
            'mileage_to_km': 1.60934,  # Miles to KM conversion
            'import_duty': 1.25,  # 25% import duty for used cars
            'currency_rate': 103.5,  # USD to KES (adjust as needed)
        }
        
    def convert_to_kenyan_market(self, df):
        """Convert US dataset to Kenyan market context"""
        df_kenya = df.copy()
        
        # 1. Price conversion to KES with Kenyan market adjustments
        df_kenya['price_kes'] = (df_kenya['price'] * self.kenyan_factors['price_adjustment'] * 
                               self.kenyan_factors['import_duty'] * self.kenyan_factors['currency_rate'])
        
        # 2. Convert mileage from miles to kilometers
        df_kenya['odometer_km'] = df_kenya['odometer'] * self.kenyan_factors['mileage_to_km']
        
        # 3. Calculate car age
        current_year = 2024
        df_kenya['car_age'] = current_year - df_kenya['year']
        
        # 4. Kenyan market specific adjustments
        df_kenya = self._apply_kenyan_market_rules(df_kenya)
        
        return df_kenya
    
    def _apply_kenyan_market_rules(self, df):
        """Apply Kenyan-specific market rules and preferences"""
        
        # Kenyan market prefers certain brands
        popular_brands_kenya = ['toyota', 'nissan', 'subaru', 'mitsubishi', 'mercedes-benz', 'bmw']
        df['is_popular_brand'] = df['manufacturer'].str.lower().isin(popular_brands_kenya)
        
        # Adjust prices based on brand popularity
        df.loc[df['is_popular_brand'], 'price_kes'] *= 1.1
        
        # Kenyan road conditions adjustment (higher mileage depreciation)
        df['condition_factor'] = np.where(df['odometer_km'] > 100000, 0.8, 1.0)
        df['price_kes'] *= df['condition_factor']
        
        # Fuel type preferences in Kenya
        fuel_preference = {
            'gas': 1.0,
            'diesel': 1.15,  # Diesel preferred for fuel efficiency
            'hybrid': 1.2,   # Growing popularity
            'electric': 0.8  # Limited infrastructure
        }
        
        df['fuel_premium'] = df['fuel'].map(fuel_preference).fillna(1.0)
        df['price_kes'] *= df['fuel_premium']
        
        return df

class DataPreprocessor:
    def __init__(self):
        self.converter = KenyanMarketConverter()
        self.label_encoders = {}
        
    def load_and_clean_data(self, file_path):
        """Load and clean the raw dataset"""
        df = pd.read_csv(file_path)
        
        # Basic cleaning
        df = self._clean_data(df)
        
        # Convert to Kenyan market
        df_kenya = self.converter.convert_to_kenyan_market(df)
        
        return df_kenya
    
    def _clean_data(self, df):
        """Clean the raw dataset"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df['odometer'] = df['odometer'].fillna(df['odometer'].median())
        df['year'] = df['year'].fillna(df['year'].median())
        
        # Remove unrealistic prices and years
        df = df[(df['price'] > 500) & (df['price'] < 100000)]
        df = df[df['year'] > 1990]
        
        # Clean manufacturer names
        df['manufacturer'] = df['manufacturer'].str.lower().str.strip()
        
        return df
    
    def engineer_features(self, df):
        """Create additional features for the Kenyan market"""
        # Vehicle age categories
        df['age_category'] = pd.cut(df['car_age'], 
                                  bins=[0, 3, 7, 15, 100],
                                  labels=['New', 'Young', 'Middle', 'Old'])
        
        # Mileage categories
        df['mileage_category'] = pd.cut(df['odometer_km'],
                                      bins=[0, 50000, 100000, 200000, 1000000],
                                      labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Price categories for Kenyan market
        df['price_category'] = pd.cut(df['price_kes'],
                                    bins=[0, 500000, 1000000, 3000000, 5000000, 100000000],
                                    labels=['Budget', 'Affordable', 'Mid-range', 'Luxury', 'Premium'])
        
        return df
    
    def prepare_for_training(self, df):
        """Prepare final dataset for model training"""
        # Select relevant features for Kenyan market
        features = [
            'manufacturer', 'model', 'year', 'car_age', 'odometer_km',
            'fuel', 'transmission', 'cylinders', 'drive', 'type',
            'is_popular_brand', 'age_category', 'mileage_category'
        ]
        
        # Filter and encode categorical variables
        df_final = df[features + ['price_kes']].copy()
        df_final = df_final.dropna()
        
        # Encode categorical variables
        categorical_cols = ['manufacturer', 'model', 'fuel', 'transmission', 
                          'drive', 'type', 'age_category', 'mileage_category']
        
        for col in categorical_cols:
            if col in df_final.columns:
                le = LabelEncoder()
                df_final[col] = le.fit_transform(df_final[col].astype(str))
                self.label_encoders[col] = le
        
        return df_final