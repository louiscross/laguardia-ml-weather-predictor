from flask import Flask, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import threading
import time
import requests
from bs4 import BeautifulSoup
import logging
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///weather.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class WeatherData(db.Model):
    __tablename__ = 'weather_data'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(50), nullable=False)
    temp = db.Column(db.Float, nullable=False)
    dewpoint = db.Column(db.Float)
    humidity = db.Column(db.Float)
    wind_dir = db.Column(db.Float)
    wind_speed = db.Column(db.Float)
    visibility = db.Column(db.Float)
    pressure = db.Column(db.Float)
    conditions = db.Column(db.String(100))

def fetch_weather_data(app):
    with app.app_context():
        while True:
            try:
                # Get weather data from API
                url = "https://api.weather.gov/stations/KLGA/observations"
                headers = {
                    'User-Agent': '(myweatherapp.com, contact@myweatherapp.com)',
                    'Accept': 'application/json'
                }
                response = requests.get(url, headers=headers)
                data = response.json()

                if 'features' in data:
                    # Process each observation
                    for obs in data['features']:
                        props = obs['properties']
                        
                        # Parse timestamp
                        timestamp = datetime.fromisoformat(props['timestamp'].replace('Z', '+00:00'))
                        timestamp = timestamp.astimezone().replace(tzinfo=None)  # Convert to local time
                        
                        # Extract and convert values
                        temp = float(props['temperature']['value'] * 9/5 + 32) if props['temperature']['value'] is not None else None
                        dewpoint = float(props['dewpoint']['value'] * 9/5 + 32) if props['dewpoint']['value'] is not None else None
                        humidity = float(props['relativeHumidity']['value']) if props['relativeHumidity']['value'] is not None else None
                        wind_dir = float(props['windDirection']['value']) if props['windDirection']['value'] is not None else None
                        wind_speed = float(props['windSpeed']['value'] * 2.237) if props['windSpeed']['value'] is not None else None
                        visibility = float(props['visibility']['value'] * 0.000621371) if props['visibility']['value'] is not None else None
                        pressure = float(props['barometricPressure']['value'] / 100) if props['barometricPressure']['value'] is not None else None
                        conditions = props['textDescription']
                        
                        # Create new weather record
                        weather_data = WeatherData(
                            timestamp=timestamp.strftime('%Y-%m-%d %H:%M'),
                            temp=temp,
                            dewpoint=dewpoint,
                            humidity=humidity,
                            wind_dir=wind_dir,
                            wind_speed=wind_speed,
                            visibility=visibility,
                            pressure=pressure,
                            conditions=conditions
                        )
                        
                        # Add to database
                        try:
                            db.session.add(weather_data)
                            db.session.commit()
                            logging.info(f"Weather data added: {temp}°F, {humidity}% humidity")
                        except Exception as e:
                            logging.error(f"Error saving weather data: {str(e)}")
                            db.session.rollback()
                
            except Exception as e:
                logging.error(f"Error fetching weather data: {str(e)}")
                
            time.sleep(300)  # Wait 5 minutes before next update

def predict_weather(historical_data, hours_ahead=24):
    try:
        # Get current hour
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Only predict remaining hours of today
        hours_to_predict = 24 - current_hour
        
        if not historical_data or hours_to_predict <= 0:
            return []
            
        # Prepare training data
        X = []  # Hours of the day (0-23)
        y_temp = []  # Temperature values
        y_humidity = []  # Humidity values
        y_wind = []  # Wind speed values
        
        # Extract hour and values from historical data
        for record in historical_data:
            hour = datetime.strptime(record.timestamp, '%Y-%m-%d %H:%M').hour
            if record.temp is not None:
                X.append([hour])
                y_temp.append(record.temp)
            if record.humidity is not None:
                y_humidity.append(record.humidity)
            if record.wind_speed is not None:
                y_wind.append(record.wind_speed)
        
        if not X:  # No valid data points
            return []
            
        X = np.array(X)
        y_temp = np.array(y_temp)
        y_humidity = np.array(y_humidity)
        y_wind = np.array(y_wind)
        
        # Train models
        temp_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        humidity_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        wind_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        
        temp_model.fit(X, y_temp)
        humidity_model.fit(X, y_humidity)
        wind_model.fit(X, y_wind)
        
        # Generate predictions for remaining hours
        predictions = []
        for hour in range(current_hour + 1, 24):
            # Make predictions
            temp_pred = temp_model.predict([[hour]])[0]
            humidity_pred = humidity_model.predict([[hour]])[0]
            wind_pred = wind_model.predict([[hour]])[0]
            
            # Create timestamp for this hour
            timestamp = current_time.replace(hour=hour, minute=0).strftime('%Y-%m-%d %H:%M')
            
            predictions.append({
                'timestamp': timestamp,
                'temp': round(temp_pred, 1),
                'humidity': round(humidity_pred, 1),
                'wind_speed': round(wind_pred, 1)
            })
        
        return predictions
    except Exception as e:
        logging.error(f"Error in predict_weather: {str(e)}")
        return []

@app.route('/')
def index():
    try:
        # Get all records ordered by timestamp
        records = WeatherData.query.order_by(WeatherData.timestamp.desc()).all()
        
        # Get today's date
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Filter records for today and last 7 days
        todays_records = [r for r in records if r.timestamp.startswith(today)]
        last_7_days = [r for r in records if (datetime.now() - datetime.strptime(r.timestamp, '%Y-%m-%d %H:%M')).days <= 7]
        
        # Generate predictions for remaining hours of today
        predictions = predict_weather(last_7_days)
        
        # Prepare data for historical chart
        timestamps = [r.timestamp for r in records]
        temperatures = [r.temp for r in records]
        humidity_values = [r.humidity if r.humidity else 0 for r in records]
        wind_speeds = [r.wind_speed if r.wind_speed else 0 for r in records]
        
        # Prepare today's actual data
        today_timestamps = [r.timestamp for r in todays_records]
        today_temperatures = [r.temp for r in todays_records]
        today_humidity = [r.humidity if r.humidity else 0 for r in todays_records]
        today_wind = [r.wind_speed if r.wind_speed else 0 for r in todays_records]
        
        # Add predictions for remaining hours
        if predictions:
            predicted_timestamps = [p['timestamp'] for p in predictions]
            predicted_temperatures = [p['temp'] for p in predictions]
            predicted_humidity = [p['humidity'] for p in predictions]
            predicted_wind = [p['wind_speed'] for p in predictions]
            
            # Combine actual and predicted data
            today_timestamps.extend(predicted_timestamps)
            today_temperatures.extend(predicted_temperatures)
            today_humidity.extend(predicted_humidity)
            today_wind.extend(predicted_wind)
        
        return render_template('index.html',
                             records=records,
                             timestamps=timestamps,
                             temperatures=temperatures,
                             humidity_values=humidity_values,
                             wind_speeds=wind_speeds,
                             today_timestamps=today_timestamps,
                             today_temperatures=today_temperatures,
                             today_humidity=today_humidity,
                             today_wind=today_wind,
                             predictions=predictions,
                             datetime=datetime)
    except Exception as e:
        logging.error(f"Error in index route: {str(e)}")
        return f"An error occurred: {str(e)}", 500

@app.route('/data')
def get_data():
    records = WeatherData.query.order_by(WeatherData.timestamp.desc()).all()
    
    # Convert to list of dicts
    formatted_data = []
    for record in records:
        formatted_data.append({
            'timestamp': record.timestamp,
            'temp': record.temp,
            'dewpoint': record.dewpoint,
            'humidity': record.humidity,
            'wind_dir': record.wind_dir,
            'wind_speed': record.wind_speed,
            'visibility': record.visibility,
            'conditions': record.conditions,
            'pressure': record.pressure
        })
    
    return jsonify(formatted_data)

if __name__ == '__main__':
    # Create the database and tables
    with app.app_context():
        db.create_all()
    
    # Start background thread for data fetching
    logger.info("Starting background weather fetching thread...")
    thread = threading.Thread(target=fetch_weather_data, args=(app,), daemon=True)
    thread.start()
    
    app.run(debug=True)
