import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import sqlite3
import logging

class SleepDataCollector:
    """Collects and validates sleep-related data from users"""
    
    def __init__(self):
        self.required_fields = [
            'age', 'gender', 'bmi', 'neck_circumference', 'snoring_frequency',
            'daytime_sleepiness', 'observed_breathing_pauses', 'high_blood_pressure'
        ]
        
    def validate_input(self, data):
        """Validates user input data"""
        for field in self.required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate ranges
        if not 18 <= data['age'] <= 100:
            raise ValueError("Age must be between 18 and 100")
        if not 15 <= data['bmi'] <= 60:
            raise ValueError("BMI must be between 15 and 60")
        if not 25 <= data['neck_circumference'] <= 60:
            raise ValueError("Neck circumference must be between 25 and 60 cm")
            
    def collect_data(self):
        """Collects user data through console input"""
        data = {}
        
        data['age'] = int(input("Enter age: "))
        data['gender'] = input("Enter gender (M/F): ").upper()
        data['bmi'] = float(input("Enter BMI: "))
        data['neck_circumference'] = float(input("Enter neck circumference (cm): "))
        data['snoring_frequency'] = int(input("Enter snoring frequency (0-4): "))
        data['daytime_sleepiness'] = int(input("Enter daytime sleepiness level (0-4): "))
        data['observed_breathing_pauses'] = input("Have breathing pauses been observed? (Y/N): ").upper()
        data['high_blood_pressure'] = input("Do you have high blood pressure? (Y/N): ").upper()
        
        self.validate_input(data)
        return data

class RiskAssessmentModel:
    """Handles the sleep apnea risk assessment calculations"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generate_training_data(self):
        """Generates synthetic training data for the model"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data
        data = {
            'age': np.random.normal(50, 15, n_samples).clip(18, 100),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'bmi': np.random.normal(27, 5, n_samples).clip(15, 60),
            'neck_circumference': np.random.normal(38, 4, n_samples).clip(25, 60),
            'snoring_frequency': np.random.randint(0, 5, n_samples),
            'daytime_sleepiness': np.random.randint(0, 5, n_samples),
            'observed_breathing_pauses': np.random.choice(['Y', 'N'], n_samples),
            'high_blood_pressure': np.random.choice(['Y', 'N'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create risk scores based on medical risk factors
        risk_scores = (
            (df['bmi'] > 30).astype(int) * 0.3 +
            (df['neck_circumference'] > 40).astype(int) * 0.2 +
            (df['snoring_frequency'] >= 3).astype(int) * 0.2 +
            (df['daytime_sleepiness'] >= 3).astype(int) * 0.15 +
            (df['observed_breathing_pauses'] == 'Y').astype(int) * 0.1 +
            (df['high_blood_pressure'] == 'Y').astype(int) * 0.05
        )
        
        df['has_sleep_apnea'] = (risk_scores > 0.5).astype(int)
        return df
        
    def train_model(self):
        """Trains the model using synthetic data"""
        df = self.generate_training_data()
        
        # Prepare features
        X = df.copy()
        y = X.pop('has_sleep_apnea')
        
        # Convert categorical variables
        X['gender'] = (X['gender'] == 'M').astype(int)
        X['observed_breathing_pauses'] = (X['observed_breathing_pauses'] == 'Y').astype(int)
        X['high_blood_pressure'] = (X['high_blood_pressure'] == 'Y').astype(int)
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit the scaler and transform training data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
    def preprocess_data(self, data):
        """Preprocesses the input data for the model"""
        if not self.is_trained:
            logging.info("Training model with synthetic data...")
            self.train_model()
            
        features = pd.DataFrame([data])
        
        # Convert categorical variables
        features['gender'] = (features['gender'] == 'M').astype(int)
        features['observed_breathing_pauses'] = (features['observed_breathing_pauses'] == 'Y').astype(int)
        features['high_blood_pressure'] = (features['high_blood_pressure'] == 'Y').astype(int)
        
        return self.scaler.transform(features)
        
    def assess_risk(self, data):
        """Calculates sleep apnea risk score and category"""
        if not self.is_trained:
            logging.info("Training model with synthetic data...")
            self.train_model()
            
        processed_data = self.preprocess_data(data)
        risk_score = self.model.predict_proba(processed_data)[0][1]
        
        risk_category = "Low"
        if risk_score > 0.7:
            risk_category = "High"
        elif risk_score > 0.4:
            risk_category = "Moderate"
            
        return {
            'risk_score': risk_score,
            'risk_category': risk_category,
            'assessment_date': datetime.now()
        }

class DatabaseManager:
    """Manages database operations for storing assessment results"""
    
    def __init__(self, db_path='sleep_apnea_assessments.db'):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initializes the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_date TIMESTAMP,
                age INTEGER,
                gender TEXT,
                bmi REAL,
                neck_circumference REAL,
                snoring_frequency INTEGER,
                daytime_sleepiness INTEGER,
                observed_breathing_pauses TEXT,
                high_blood_pressure TEXT,
                risk_score REAL,
                risk_category TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def save_assessment(self, user_data, assessment_result):
        """Saves the assessment data and results to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data = {**user_data, **assessment_result}
        
        cursor.execute('''
            INSERT INTO assessments (
                assessment_date, age, gender, bmi, neck_circumference,
                snoring_frequency, daytime_sleepiness, observed_breathing_pauses,
                high_blood_pressure, risk_score, risk_category
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['assessment_date'], data['age'], data['gender'],
            data['bmi'], data['neck_circumference'], data['snoring_frequency'],
            data['daytime_sleepiness'], data['observed_breathing_pauses'],
            data['high_blood_pressure'], data['risk_score'], data['risk_category']
        ))
        
        conn.commit()
        conn.close()

class SleepApneaAssessmentSystem:
    """Main class that coordinates the sleep apnea assessment system"""
    
    def __init__(self):
        self.data_collector = SleepDataCollector()
        self.risk_model = RiskAssessmentModel()
        self.db_manager = DatabaseManager()
        logging.basicConfig(level=logging.INFO)
        logging.info("Initializing Sleep Apnea Assessment System...")
        
    def generate_recommendations(self, risk_category):
        """Generates lifestyle recommendations based on risk category"""
        recommendations = {
            "Low": [
                "Continue maintaining a healthy sleep schedule",
                "Practice good sleep hygiene",
                "Monitor any changes in sleep patterns"
            ],
            "Moderate": [
                "Consider consulting a sleep specialist",
                "Monitor and record sleep patterns",
                "Maintain a healthy weight",
                "Avoid alcohol before bedtime",
                "Sleep on your side instead of your back"
            ],
            "High": [
                "Urgent: Schedule an appointment with a sleep specialist",
                "Consider a sleep study evaluation",
                "Start keeping a detailed sleep diary",
                "Evaluate current lifestyle factors affecting sleep",
                "Implement immediate lifestyle changes (weight management, sleep position)",
                "Consider temporary use of sleep position devices"
            ]
        }
        return recommendations.get(risk_category, [])
        
    def run_assessment(self):
        """Runs the complete sleep apnea risk assessment process"""
        try:
            logging.info("Starting sleep apnea risk assessment")
            
            # Collect user data
            user_data = self.data_collector.collect_data()
            
            # Perform risk assessment
            assessment_result = self.risk_model.assess_risk(user_data)
            
            # Save to database
            self.db_manager.save_assessment(user_data, assessment_result)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(assessment_result['risk_category'])
            
            # Prepare result summary
            result_summary = {
                'risk_score': f"{assessment_result['risk_score']:.2%}",
                'risk_category': assessment_result['risk_category'],
                'recommendations': recommendations
            }
            
            return result_summary
            
        except Exception as e:
            logging.error(f"Error during assessment: {str(e)}")
            raise

if __name__ == "__main__":
    system = SleepApneaAssessmentSystem()
    try:
        print("\nSleep Apnea Risk Assessment")
        print("==========================")
        print("Please answer the following questions:")
        results = system.run_assessment()
        print("\nAssessment Results:")
        print("==================")
        print(f"Risk Score: {results['risk_score']}")
        print(f"Risk Category: {results['risk_category']}")
        print("\nRecommendations:")
        print("===============")
        for rec in results['recommendations']:
            print(f"- {rec}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")