# app.py - FINAL FIXED VERSION
from flask import Flask, render_template, request, jsonify
import random
import requests
import os
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Import config
try:
    from config import *
    print("✅ Config loaded successfully")
except ImportError:
    print("⚠️ Using default config")
    # Default config if config.py doesn't exist
    USD_TO_KES_RATE = 130.0
    MILES_TO_KM = 1.60934
    SAMPLE_SIZE = 20000
    # Add API keys if not in config
    UNSPLASH_ACCESS_KEY = os.environ.get('UNSPLASH_ACCESS_KEY', 'yPHgXJrWyINAgdbm1dP_MuFAD3yFU2fl21JMkZXtStY')
    PEXELS_API_KEY = os.environ.get('PEXELS_API_KEY', 'b9z182Xa4XOT63WilFJ3fixA13b97KikRdykyQdgsVWCQrHzXaUSnvHp')

class CarPricePredictor:
    def __init__(self):
        print("🚗 Initializing Car Price Predictor")
        
        # Currency settings
        self.usd_to_kes_rate = USD_TO_KES_RATE if 'USD_TO_KES_RATE' in globals() else 130.0
        self.miles_to_km = MILES_TO_KM if 'MILES_TO_KM' in globals() else 1.60934
        
        # API Keys for images
        self.unsplash_access_key = UNSPLASH_ACCESS_KEY if 'UNSPLASH_ACCESS_KEY' in globals() else os.environ.get('UNSPLASH_ACCESS_KEY', '')
        self.pexels_api_key = PEXELS_API_KEY if 'PEXELS_API_KEY' in globals() else os.environ.get('PEXELS_API_KEY', '')
        
        # Initialize model and data
        self.model = None
        self.data = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = []
        self.numerical_columns = []
        self.model_loaded = False
        
        # Load or train model
        self._initialize_model()
        
        print("✅ Car Price Predictor initialized successfully!")
    
    def _initialize_model(self):
        """Load or train the ML model"""
        model_path = 'models/kenyan_car_price_model.pkl'
        
        try:
            # Try to load existing model
            if os.path.exists(model_path):
                print(f"📂 Loading pre-trained model from {model_path}")
                
                # Load model components
                if os.path.exists('models/kenyan_car_price_model.pkl'):
                    self.model = joblib.load('models/kenyan_car_price_model.pkl')
                    print("✅ Model loaded")
                
                if os.path.exists('models/feature_columns.pkl'):
                    self.feature_columns = joblib.load('models/feature_columns.pkl')
                    print(f"📊 Feature columns loaded: {self.feature_columns}")
                
                if os.path.exists('models/label_encoders.pkl'):
                    self.label_encoders = joblib.load('models/label_encoders.pkl')
                    print(f"🔤 Label encoders loaded: {len(self.label_encoders)} categorical columns")
                
                if os.path.exists('models/scaler.pkl'):
                    self.scaler = joblib.load('models/scaler.pkl')
                    print(f"⚖️ Scaler loaded")
                
                # Identify numerical columns
                self.numerical_columns = ['year', 'odometer_km', 'car_age']
                self.numerical_columns = [col for col in self.numerical_columns if col in self.feature_columns]
                
                self.model_loaded = True
                print("✅ Model loaded successfully")
                
            else:
                print("🔄 No pre-trained model found. Using rule-based prediction.")
                self.model_loaded = False
                
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            print("🔄 Using rule-based prediction")
            self.model_loaded = False
    
    def predict(self, data):
        """Make a prediction"""
        print(f"\n📋 Prediction Request:")
        for key, value in data.items():
            print(f"  {key}: {value}")
        
        # Try ML prediction if model is loaded
        if self.model_loaded and self.model is not None:
            print("🤖 Using ML model for prediction")
            try:
                return self._ml_prediction(data), "ml_model"
            except Exception as e:
                print(f"⚠️ ML prediction failed: {e}")
                print("📊 Falling back to rule-based prediction")
                return self._rule_based_prediction(data), "rule_based"
        
        # Use rule-based
        print("📊 Using rule-based prediction")
        return self._rule_based_prediction(data), "rule_based"
    
    def _ml_prediction(self, data):
        """Make ML prediction - COMPLETELY FIXED"""
        print("🔍 Preparing features for ML model...")
        
        # Get current year for car age calculation
        current_year = datetime.now().year
        
        # Create features dictionary
        features_dict = {
            'year': int(data.get('year', 2018)),
            'odometer_km': float(data.get('odometer', 50000)),
            'car_age': current_year - int(data.get('year', 2018)),
            'manufacturer': str(data.get('manufacturer', 'toyota')).lower(),
            'model': str(data.get('model', 'corolla')).lower(),
            'condition': str(data.get('condition', 'good')).lower(),
            'fuel': str(data.get('fuel', 'gas')).lower(),
            'transmission': str(data.get('transmission', 'automatic')).lower(),
            'drive': str(data.get('drive', 'fwd')).lower(),
            'type': str(data.get('type', 'sedan')).lower()
        }
        
        print(f"📋 Features prepared: {features_dict}")
        
        # Create DataFrame with all features in correct order
        df = pd.DataFrame([features_dict])
        
        # Reorder columns to match training data
        if self.feature_columns:
            # Ensure we have all columns
            for col in self.feature_columns:
                if col not in df.columns:
                    print(f"⚠️ Adding missing column: {col}")
                    if col in ['year', 'odometer_km', 'car_age']:
                        df[col] = 0
                    else:
                        df[col] = 'unknown'
            
            # Reorder columns to match training
            df = df[self.feature_columns]
        
        print(f"📋 DataFrame columns: {list(df.columns)}")
        print(f"📋 DataFrame shape: {df.shape}")
        
        # Create a copy for processing
        df_processed = df.copy()
        
        # STEP 1: First encode all categorical features
        for col in df_processed.columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                value = str(df_processed[col].iloc[0])
                
                if value in le.classes_:
                    df_processed[col] = le.transform([value])[0]
                    print(f"✅ Encoded {col}: '{value}' -> {df_processed[col].iloc[0]}")
                else:
                    # Use most common class for unseen labels
                    df_processed[col] = le.transform([le.classes_[0]])[0]
                    print(f"⚠️ Unseen {col} '{value}', using '{le.classes_[0]}' -> {df_processed[col].iloc[0]}")
        
        # STEP 2: Prepare data for scaling
        # The scaler expects the FULL feature matrix with all columns
        # Convert to numpy array
        X_array = df_processed.values
        
        print(f"📋 Array shape before scaling: {X_array.shape}")
        
        # STEP 3: Scale using the trained scaler
        if self.scaler:
            X_scaled = self.scaler.transform(X_array)
            print(f"✅ Scaled successfully")
        else:
            X_scaled = X_array
        
        # Ensure we have the right shape
        X_final = X_scaled.reshape(1, -1)
        
        print(f"📋 Final input shape: {X_final.shape}")
        
        # STEP 4: Make prediction
        prediction = self.model.predict(X_final)[0]
        price_usd = float(prediction)
        
        # Ensure reasonable price
        price_usd = max(100, min(price_usd, 500000))
        
        print(f"✅ ML prediction successful: ${price_usd:,.2f}")
        return price_usd
    
    def _rule_based_prediction(self, data):
        """Fallback rule-based prediction"""
        print("📊 Using rule-based prediction")
        
        base_price = 500000
        year = int(data.get('year', 2018))
        mileage_km = float(data.get('odometer', 50000))
        manufacturer = data.get('manufacturer', 'toyota').lower()
        fuel = data.get('fuel', 'gas')
        transmission = data.get('transmission', 'automatic')
        condition = data.get('condition', 'good')
        
        # Adjust for Kenyan market preferences
        year_bonus = (year - 2000) * 50000
        mileage_penalty = max(0, mileage_km - 50000) * 2
        
        # Brand factors
        brand_factors = {
            'toyota': 1.3, 'subaru': 1.2, 'nissan': 1.1, 'mitsubishi': 1.1,
            'honda': 1.0, 'mazda': 0.9, 'ford': 0.8, 'mercedes-benz': 1.4,
            'bmw': 1.4, 'audi': 1.3, 'volkswagen': 1.0, 'isuzu': 1.1
        }
        brand_factor = brand_factors.get(manufacturer, 0.8)
        
        # Fuel factors
        fuel_factors = {'diesel': 1.2, 'gas': 1.0, 'petrol': 1.0, 'hybrid': 1.3, 'electric': 0.8}
        fuel_factor = fuel_factors.get(fuel, 1.0)
        
        # Transmission factor
        transmission_factor = 1.1 if transmission == 'automatic' else 1.0
        
        # Condition factor - FIXED: Added missing closing brace
        condition_factors = {
            'excellent': 1.2,
            'good': 1.0,
            'fair': 0.8,
            'poor': 0.6,
            'salvage': 0.4
        }  # This was missing
        condition_factor = condition_factors.get(condition, 1.0)
        
        # Calculate price in KES
        price_kes = (base_price + year_bonus - mileage_penalty) * brand_factor * fuel_factor * transmission_factor * condition_factor
        price_kes = max(100000, min(price_kes, 10000000))
        
        # Convert to USD
        price_usd = price_kes / self.usd_to_kes_rate
        
        print(f"✅ Rule-based prediction: KSh {price_kes:,.0f} (${price_usd:,.2f})")
        return price_usd
    
    def get_market_insights(self, data, price_kes):
        """Get market insights"""
        year = int(data.get('year', 2018))
        manufacturer = data.get('manufacturer', 'toyota').lower()
        vehicle_type = data.get('type', 'sedan').lower()
        
        insights = {
            'demand_level': 'High' if manufacturer in ['toyota', 'honda', 'subaru'] else 'Medium',
            'resale_value': 'Excellent' if year >= 2018 else 'Good',
            'maintenance_cost': 'Low' if manufacturer in ['toyota', 'honda'] else 'Medium',
            'fuel_efficiency': 'Very Good' if data.get('fuel') in ['diesel', 'hybrid'] else 'Good'
        }
        
        if vehicle_type == 'pickup':
            insights['demand_level'] = 'Very High'
            insights['resale_value'] = 'Excellent'
        
        return insights
    
    def get_price_analysis(self, data, price_kes):
        """Get price analysis"""
        model = data.get('model', 'corolla').lower()
        year = data.get('year', 2018)
        
        analysis = {
            'price_justification': f'Based on {year} {model.title()} with {float(data.get("odometer", 50000)):,.0f} km',
            'market_position': 'Competitively Priced',
            'depreciation_rate': 'Standard (15-20% per year)',
            'negotiation_margin': '5-10% below asking price'
        }
        
        if price_kes > 2500000:
            analysis['market_position'] = 'Premium Segment'
        elif price_kes < 800000:
            analysis['market_position'] = 'Budget Segment'
        
        return analysis
    
    def fetch_car_images(self, query, count=6, use_unsplash=True, use_pexels=True):
        """Fetch car images from Unsplash and Pexels APIs"""
        images = []
        
        if use_unsplash and self.unsplash_access_key and self.unsplash_access_key != 'your_unsplash_access_key':
            try:
                print(f"🌅 Fetching images from Unsplash for: {query}")
                unsplash_url = "https://api.unsplash.com/search/photos"
                params = {
                    'query': query + ' car',
                    'per_page': min(count, 30),
                    'orientation': 'landscape',
                    'client_id': self.unsplash_access_key
                }
                
                response = requests.get(unsplash_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for photo in data.get('results', [])[:count//2]:
                        images.append({
                            'url': photo['urls']['regular'],
                            'thumbnail': photo['urls']['small'],
                            'photographer': photo['user']['name'],
                            'source': 'Unsplash',
                            'alt': photo.get('alt_description', query)
                        })
                    print(f"✅ Found {len(images)} images from Unsplash")
                else:
                    print(f"⚠️ Unsplash API error: {response.status_code}")
            except Exception as e:
                print(f"⚠️ Error fetching from Unsplash: {e}")
        
        if use_pexels and self.pexels_api_key and self.pexels_api_key != 'your_pexels_api_key':
            try:
                print(f"🌅 Fetching images from Pexels for: {query}")
                pexels_url = "https://api.pexels.com/v1/search"
                headers = {
                    'Authorization': self.pexels_api_key
                }
                params = {
                    'query': query + ' car',
                    'per_page': min(count, 30),
                    'orientation': 'landscape'
                }
                
                response = requests.get(pexels_url, headers=headers, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for photo in data.get('photos', [])[:count//2]:
                        images.append({
                            'url': photo['src']['large'],
                            'thumbnail': photo['src']['medium'],
                            'photographer': photo['photographer'],
                            'source': 'Pexels',
                            'alt': query
                        })
                    print(f"✅ Found {len(images)} images from Pexels")
                else:
                    print(f"⚠️ Pexels API error: {response.status_code}")
            except Exception as e:
                print(f"⚠️ Error fetching from Pexels: {e}")
        
        # If no API keys or APIs failed, use placeholder images
        if not images:
            print("⚠️ No API keys or APIs failed, using placeholder images")
            for i in range(min(count, 4)):
                images.append({
                    'url': f'https://picsum.photos/800/600?random={i}&car={query.replace(" ", "+")}',
                    'thumbnail': f'https://picsum.photos/400/300?random={i}&car={query.replace(" ", "+")}',
                    'photographer': 'Placeholder',
                    'source': 'Lorem Picsum',
                    'alt': f'{query} car image'
                })
        
        # Shuffle and limit images
        random.shuffle(images)
        return images[:count]

# Initialize predictor
predictor = CarPricePredictor()

# Routes
@app.route('/')
def landing():
    """Render the landing page"""
    return render_template('landing.html')

@app.route('/predict-page')
def predict_page():
    """Render the prediction form page"""
    return render_template('index.html', model_loaded=predictor.model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = {
            'manufacturer': request.form.get('manufacturer', 'toyota').lower(),
            'model': request.form.get('model', 'corolla').lower(),
            'year': int(request.form.get('year', 2018)),
            'odometer': float(request.form.get('odometer_km', 50000)),
            'condition': request.form.get('condition', 'good').lower(),
            'fuel': request.form.get('fuel', 'gas').lower(),
            'transmission': request.form.get('transmission', 'automatic').lower(),
            'cylinders': request.form.get('cylinders', '4'),
            'drive': request.form.get('drive', 'fwd').lower(),
            'type': request.form.get('type', 'sedan').lower(),
        }
        
        # Get image options
        generate_image = request.form.get('generate_image') == 'true'
        search_images = request.form.get('search_images') == 'true'
        
        # Create image query
        image_query = f"{data['year']} {data['manufacturer'].title()} {data['model'].title()} {data['type'].title()}"
        
        # Make prediction
        price_usd, prediction_type = predictor.predict(data)
        price_kes = price_usd * predictor.usd_to_kes_rate
        
        # Format prices
        formatted_price_usd = f"${price_usd:,.2f}"
        formatted_price_kes = f"KSh {price_kes:,.0f}"
        
        # Determine confidence
        confidence = 85 if prediction_type == 'ml_model' else 75
        
        # Get market insights
        market_insights = predictor.get_market_insights(data, price_kes)
        
        # Get price analysis
        price_analysis = predictor.get_price_analysis(data, price_kes)
        
        # Fetch car images if requested
        car_images = []
        if generate_image or search_images:
            car_images = predictor.fetch_car_images(
                query=image_query,
                count=6,
                use_unsplash=True,
                use_pexels=True
            )
        
        # Prepare vehicle info for template
        vehicle_info = {
            'make': data['manufacturer'].title(),
            'model': data['model'].title(),
            'year': data['year'],
            'mileage': f"{data['odometer']:,.0f} km",
            'fuel_type': data['fuel'].title(),
            'transmission': data['transmission'].title(),
            'vehicle_type': data['type'].title(),
            'condition': data['condition'].title(),
            'drive_type': data['drive'].upper(),
            'engine_size': data['cylinders']
        }
        
        # Prepare specifications
        specifications = {
            'engine': f"{data['cylinders']} Cylinder",
            'drive_type': data['drive'].upper(),
            'vehicle_type': data['type'].title(),
            'fuel_efficiency': 'Good' if data['fuel'] in ['diesel', 'hybrid'] else 'Average',
            'cylinders': data['cylinders']
        }
        
        # Prepare recommendations
        recommendations = [
            "Consider a pre-purchase inspection from a trusted mechanic",
            "Verify service history and maintenance records",
            "Test drive the vehicle on different road conditions",
            "Compare prices with similar models in the market",
            "Check for any outstanding loans on the vehicle"
        ]
        
        if int(data['year']) < 2015:
            recommendations.append("Consider comprehensive mechanical inspection due to vehicle age")
        
        if float(data['odometer']) > 150000:
            recommendations.append("Check timing belt and major service history")
        
        print(f"\n✅ Prediction Complete:")
        print(f"  Price USD: {formatted_price_usd}")
        print(f"  Price KES: {formatted_price_kes}")
        print(f"  Prediction type: {prediction_type}")
        print(f"  Images fetched: {len(car_images)}")
        
        # Render result template with all data
        return render_template('result.html',
            price_kes=formatted_price_kes,
            price_usd=formatted_price_usd,
            confidence=confidence,
            vehicle_info=vehicle_info,
            specifications=specifications,
            market_insights=market_insights,
            price_analysis=price_analysis,
            recommendations=recommendations,
            prediction_type=prediction_type,
            image_query=image_query,
            generate_image=generate_image,
            search_images=search_images,
            car_images=car_images,
            model_loaded=predictor.model_loaded)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        
        return render_template('error.html',
            error=str(e),
            message='An error occurred during prediction'
        )

@app.route('/api/fetch-images', methods=['POST'])
def fetch_images_api():
    """API endpoint to fetch car images"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        count = data.get('count', 6)
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        images = predictor.fetch_car_images(
            query=query,
            count=min(count, 12),
            use_unsplash=True,
            use_pexels=True
        )
        
        return jsonify({
            'success': True,
            'query': query,
            'images': images,
            'count': len(images)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor.model_loaded,
        'usd_to_kes_rate': predictor.usd_to_kes_rate,
        'apis_configured': {
            'unsplash': bool(predictor.unsplash_access_key and predictor.unsplash_access_key != 'your_unsplash_access_key'),
            'pexels': bool(predictor.pexels_api_key and predictor.pexels_api_key != 'your_pexels_api_key')
        }
    })

if __name__ == '__main__':
    print("🚗 Starting Kenyan Car Price Predictor")
    print("🌐 Access: http://localhost:5000")
    print(f"💰 Exchange rate: 1 USD = {predictor.usd_to_kes_rate} KES")
    print(f"🤖 Model: {'ML Model Loaded' if predictor.model_loaded else 'Rule-Based Prediction'}")
    print(f"🖼️  Image APIs: Unsplash: {'✓' if predictor.unsplash_access_key and predictor.unsplash_access_key != 'your_unsplash_access_key' else '✗'}, Pexels: {'✓' if predictor.pexels_api_key and predictor.pexels_api_key != 'your_pexels_api_key' else '✗'}")
    
    print(f"\n🔗 Available endpoints:")
    print(f"  http://localhost:5000/ - Landing page")
    print(f"  http://localhost:5000/predict-page - Prediction form")
    print(f"  http://localhost:5000/predict - Make prediction")
    print(f"  http://localhost:5000/api/fetch-images - Fetch images API")
    print(f"  http://localhost:5000/health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
    if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)