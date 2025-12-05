# src/prediction.py - Enhanced with car database and market insights
from car_database import CarDatabase

class KenyanCarPricePredictor:
    def __init__(self, model_path=None):
        self.kenyan_conversion = {
            'USD_TO_KES': 103.5,
            'PRICE_ADJUSTMENT': 0.3,
            'IMPORT_DUTY': 1.25
        }
        self.model = None
        self.car_db = CarDatabase()  # Add car database
        
        if model_path:
            try:
                import joblib
                self.model = joblib.load(model_path)
                print("✓ ML Model loaded successfully")
            except Exception as e:
                print(f"✗ Could not load ML model: {e}")
                print("Using rule-based prediction instead")
    
    def predict_price(self, input_features):
        """Predict car price for Kenyan market with enhanced details"""
        try:
            if self.model:
                prediction = self._predict_with_model(input_features)
            else:
                prediction = self._predict_rule_based(input_features)
            
            # Add car specifications and image - UPDATED: Pass year parameter
            car_details = self.car_db.get_car_details(
                input_features.get('manufacturer', ''),
                input_features.get('model', ''),
                input_features.get('year')  # Pass the year for better image search
            )
            
            # Add market insights and analysis
            prediction.update({
                'car_image': car_details['image'],
                'specifications': car_details['specs'],
                'market_insights': self._get_market_insights(input_features),
                'price_analysis': self._analyze_price(prediction['price_kes'], input_features),
                'recommendations': self._get_recommendations(input_features, prediction['price_kes'])
            })
            
            return prediction
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return self._simple_fallback(input_features)
    
    def _predict_with_model(self, input_features):
        """Predict using trained ML model"""
        # This would need feature encoding to match training data
        # For now, use rule-based
        return self._predict_rule_based(input_features)
    
    def _predict_rule_based(self, input_features):
        """Rule-based price prediction for Kenyan market"""
        manufacturer = input_features.get('manufacturer', '').lower()
        year = input_features.get('year', 2015)
        mileage_km = input_features.get('odometer_km', 50000)
        fuel_type = input_features.get('fuel', 'petrol')
        transmission = input_features.get('transmission', 'automatic')
        cylinders = input_features.get('cylinders', '4')
        drive_type = input_features.get('drive', 'fwd')
        vehicle_type = input_features.get('type', 'sedan')
        
        # Base price calculation
        base_price = 300000  # Base price in KES
        
        # Brand adjustments (Toyota is most popular in Kenya)
        brand_factors = {
            'toyota': 1.3,
            'subaru': 1.2,
            'nissan': 1.1,
            'mitsubishi': 1.1,
            'honda': 1.0,
            'mazda': 0.9,
            'ford': 0.8,
            'mercedes-benz': 1.5,
            'bmw': 1.5,
            'audi': 1.4,
            'volkswagen': 1.0,
            'isuzu': 1.2,
            'chevrolet': 0.8
        }
        brand_factor = brand_factors.get(manufacturer, 0.7)
        
        # Year adjustment
        year_factor = max(0, (year - 2000)) * 40000
        
        # Mileage adjustment
        mileage_factor = max(0, (150000 - mileage_km)) * 5
        
        # Fuel type adjustment
        fuel_factors = {
            'diesel': 1.2,    # Diesel preferred for fuel efficiency in Kenya
            'petrol': 1.0,    # Standard petrol
            'hybrid': 1.3,    # Hybrids have premium pricing
            'electric': 0.8   # Limited infrastructure reduces value
        }
        fuel_factor = fuel_factors.get(fuel_type, 1.0)
        
        # Transmission adjustment
        transmission_factor = 1.1 if transmission == 'automatic' else 1.0
        
        # Engine size (cylinders) adjustment
        cylinder_factors = {
            '4': 1.0,   # Standard 4-cylinder
            '6': 1.3,   # 6-cylinder premium
            '8': 1.6,   # V8 luxury
            '10': 2.0,  # High-performance
            '12': 2.5   # Luxury/sports
        }
        cylinder_factor = cylinder_factors.get(cylinders, 1.0)
        
        # Drive type adjustment
        drive_factors = {
            'fwd': 1.0,   # Front-wheel drive (standard)
            'rwd': 1.1,   # Rear-wheel drive (sporty)
            'awd': 1.3,   # All-wheel drive (premium)
            '4wd': 1.4    # 4-wheel drive (off-road capability)
        }
        drive_factor = drive_factors.get(drive_type, 1.0)
        
        # Vehicle type adjustment
        type_factors = {
            'sedan': 1.0,        # Standard sedan
            'suv': 1.4,          # SUVs are popular in Kenya
            'truck': 1.5,        # Trucks have high utility value
            'hatchback': 0.9,    # Hatchbacks are economical
            'coupe': 1.2,        # Coupes are sporty
            'convertible': 1.3,  # Convertibles are luxury
            'wagon': 1.0,        # Station wagons
            'minivan': 1.1,      # Family vehicles
            'pickup': 1.6        # Pickups are very popular in Kenya
        }
        type_factor = type_factors.get(vehicle_type, 1.0)
        
        # Calculate final price with all factors
        price_kes = (base_price + year_factor + mileage_factor) * \
                   brand_factor * fuel_factor * transmission_factor * \
                   cylinder_factor * drive_factor * type_factor
        
        # Ensure reasonable range for Kenyan market
        price_kes = max(100000, min(price_kes, 10000000))  # 100K to 10M KES range
        
        # Calculate confidence based on input completeness
        confidence = self._calculate_confidence(input_features)
        
        return {
            'price_kes': round(price_kes),
            'price_usd': round(price_kes / self.kenyan_conversion['USD_TO_KES']),
            'confidence': confidence
        }
    
    def _get_market_insights(self, input_features):
        """Generate market insights for the car"""
        manufacturer = input_features.get('manufacturer', '').lower()
        year = input_features.get('year', 2015)
        mileage_km = input_features.get('odometer_km', 50000)
        vehicle_type = input_features.get('type', 'sedan')
        
        current_year = 2024
        car_age = current_year - year
        
        insights = {
            'popularity': self._get_popularity_level(manufacturer),
            'resale_value': self._get_resale_value(manufacturer, car_age),
            'maintenance_cost': self._get_maintenance_cost(manufacturer),
            'market_demand': self._get_market_demand(vehicle_type, year),
            'fuel_efficiency': self._get_fuel_efficiency_rating(mileage_km, vehicle_type),
            'reliability': self._get_reliability_rating(manufacturer)
        }
        
        return insights
    
    def _get_popularity_level(self, manufacturer):
        """Determine popularity level of the brand in Kenya"""
        popular_brands = ['toyota', 'subaru', 'nissan']
        medium_brands = ['mitsubishi', 'honda', 'mazda', 'isuzu']
        
        if manufacturer in popular_brands:
            return 'High'
        elif manufacturer in medium_brands:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_resale_value(self, manufacturer, car_age):
        """Determine resale value rating"""
        if manufacturer == 'toyota':
            return 'Excellent' if car_age <= 5 else 'Very Good'
        elif manufacturer in ['subaru', 'honda']:
            return 'Very Good' if car_age <= 5 else 'Good'
        else:
            return 'Good' if car_age <= 3 else 'Average'
    
    def _get_maintenance_cost(self, manufacturer):
        """Determine maintenance cost level"""
        low_cost_brands = ['toyota', 'honda', 'nissan']
        medium_cost_brands = ['subaru', 'mitsubishi', 'mazda']
        
        if manufacturer in low_cost_brands:
            return 'Low'
        elif manufacturer in medium_cost_brands:
            return 'Medium'
        else:
            return 'High'
    
    def _get_market_demand(self, vehicle_type, year):
        """Determine market demand level"""
        current_year = 2024
        car_age = current_year - year
        
        if vehicle_type in ['suv', 'pickup'] and car_age <= 5:
            return 'Very High'
        elif vehicle_type in ['sedan', 'hatchback'] and car_age <= 3:
            return 'High'
        else:
            return 'Medium'
    
    def _get_fuel_efficiency_rating(self, mileage_km, vehicle_type):
        """Determine fuel efficiency rating"""
        if mileage_km <= 50000:
            return 'Excellent'
        elif mileage_km <= 100000:
            return 'Good'
        elif mileage_km <= 150000:
            return 'Average'
        else:
            return 'Low'
    
    def _get_reliability_rating(self, manufacturer):
        """Determine reliability rating"""
        high_reliability = ['toyota', 'honda', 'subaru']
        medium_reliability = ['nissan', 'mazda', 'mitsubishi']
        
        if manufacturer in high_reliability:
            return 'High'
        elif manufacturer in medium_reliability:
            return 'Medium'
        else:
            return 'Standard'
    
    def _analyze_price(self, price_kes, input_features):
        """Provide detailed price analysis"""
        manufacturer = input_features.get('manufacturer', '').lower()
        year = input_features.get('year', 2015)
        
        analysis = {
            'price_category': self._get_price_category(price_kes),
            'value_for_money': self._get_value_rating(price_kes, manufacturer, year),
            'depreciation_rate': self._get_depreciation_rate(year),
            'insurance_estimate': self._get_insurance_estimate(price_kes)
        }
        
        return analysis
    
    def _get_price_category(self, price_kes):
        """Categorize the price range"""
        if price_kes <= 500000:
            return 'Budget'
        elif price_kes <= 1000000:
            return 'Affordable'
        elif price_kes <= 3000000:
            return 'Mid-Range'
        elif price_kes <= 5000000:
            return 'Premium'
        else:
            return 'Luxury'
    
    def _get_value_rating(self, price_kes, manufacturer, year):
        """Determine value for money rating"""
        current_year = 2024
        car_age = current_year - year
        
        # Toyota and Honda generally offer better value
        if manufacturer in ['toyota', 'honda'] and car_age <= 5:
            return 'Excellent'
        elif manufacturer in ['subaru', 'nissan'] and car_age <= 3:
            return 'Very Good'
        else:
            return 'Good'
    
    def _get_depreciation_rate(self, year):
        """Estimate depreciation rate"""
        current_year = 2024
        car_age = current_year - year
        
        if car_age <= 2:
            return '15-20% per year'
        elif car_age <= 5:
            return '10-15% per year'
        else:
            return '5-10% per year'
    
    def _get_insurance_estimate(self, price_kes):
        """Estimate annual insurance cost"""
        # Insurance is typically 3-5% of car value in Kenya
        insurance_low = price_kes * 0.03
        insurance_high = price_kes * 0.05
        return f"KES {int(insurance_low):,} - {int(insurance_high):,} per year"
    
    def _get_recommendations(self, input_features, price_kes):
        """Provide personalized recommendations"""
        manufacturer = input_features.get('manufacturer', '').lower()
        year = input_features.get('year', 2015)
        mileage_km = input_features.get('odometer_km', 50000)
        
        recommendations = []
        current_year = 2024
        car_age = current_year - year
        
        # Price-based recommendations
        if price_kes > 3000000:
            recommendations.append("Consider comprehensive insurance for better protection")
        
        if car_age > 10:
            recommendations.append("Budget for potential major maintenance items")
        
        if mileage_km > 100000:
            recommendations.append("Check service history for timing belt and major services")
        
        # Brand-specific recommendations
        if manufacturer == 'subaru':
            recommendations.append("Regular AWD system maintenance recommended")
        elif manufacturer in ['bmw', 'mercedes-benz', 'audi']:
            recommendations.append("Consider extended warranty for European luxury cars")
        
        # General recommendations
        if car_age <= 3:
            recommendations.append("Good choice - modern safety and fuel efficiency features")
        
        if len(recommendations) == 0:
            recommendations.append("Well-maintained vehicle with good market value")
        
        return recommendations
    
    def _calculate_confidence(self, input_features):
        """Calculate prediction confidence based on input completeness"""
        required_fields = ['manufacturer', 'year', 'odometer_km', 'fuel', 'transmission']
        optional_fields = ['cylinders', 'drive', 'type']
        
        total_fields = len(required_fields) + len(optional_fields)
        filled_count = 0
        
        # Check required fields
        for field in required_fields:
            if input_features.get(field) and input_features[field] != '':
                filled_count += 1
        
        # Check optional fields (half weight)
        for field in optional_fields:
            if input_features.get(field) and input_features[field] != '':
                filled_count += 0.5
        
        # Base confidence from required fields
        base_confidence = filled_count / (len(required_fields) + len(optional_fields) * 0.5)
        
        # Adjust for data quality
        year = input_features.get('year', 0)
        mileage = input_features.get('odometer_km', 0)
        
        # Reasonable year range check
        if 1990 <= year <= 2024:
            base_confidence += 0.1
        
        # Reasonable mileage check
        if 0 <= mileage <= 500000:
            base_confidence += 0.1
        
        return min(0.95, max(0.5, base_confidence))  # Cap between 50% and 95%
    
    def _simple_fallback(self, input_features):
        """Simple fallback calculation when other methods fail"""
        # Very basic calculation as last resort
        year = input_features.get('year', 2015)
        base_price = 500000
        
        year_bonus = (year - 2000) * 30000
        fallback_price = base_price + year_bonus
        
        return {
            'price_kes': max(200000, fallback_price),
            'price_usd': round(max(200000, fallback_price) / self.kenyan_conversion['USD_TO_KES']),
            'confidence': 0.5,
            'car_image': '/static/images/default_car.jpg',
            'specifications': {
                'engine': 'Information not available',
                'horsepower': 'Information not available',
                'fuel_economy': 'Information not available',
                'seating': '5 seats',
                'safety': 'Rating not available',
                'features': ['Standard features']
            },
            'market_insights': {
                'popularity': 'Unknown',
                'resale_value': 'Unknown',
                'maintenance_cost': 'Unknown',
                'market_demand': 'Unknown'
            },
            'price_analysis': {
                'price_category': 'Unknown',
                'value_for_money': 'Unknown',
                'depreciation_rate': 'Unknown',
                'insurance_estimate': 'Unknown'
            },
            'recommendations': ['Using basic calculation method'],
            'note': 'Used basic fallback calculation due to system error'
        }

# Test the enhanced predictor
if __name__ == "__main__":
    predictor = KenyanCarPricePredictor()
    
    # Test with sample data
    test_data = {
        'manufacturer': 'toyota',
        'model': 'corolla',
        'year': 2018,
        'odometer_km': 50000,
        'fuel': 'petrol',
        'transmission': 'automatic',
        'cylinders': '4',
        'drive': 'fwd',
        'type': 'sedan'
    }
    
    result = predictor.predict_price(test_data)
    print("Enhanced Prediction Test:")
    print(f"Price: KES {result['price_kes']:,}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Car Image: {result['car_image']}")
    print("\nMarket Insights:")
    for key, value in result['market_insights'].items():
        print(f"  {key}: {value}")
    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  • {rec}")