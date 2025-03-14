# LaGuardia ML Weather Predictor

An advanced machine learning-powered weather forecasting system that fetches real-time data from LaGuardia Airport (KLGA) weather station, processes historical patterns, and generates accurate predictions using sophisticated ML models.

## Features

- Real-time weather data from LaGuardia Airport (KLGA) NOAA Weather API
- Machine learning-based weather predictions using polynomial regression
- Historical weather pattern analysis and tracking
- Interactive data visualization with real-time updates
- Automated data collection every 5 minutes
- Multi-parameter predictions (temperature, humidity, wind speed)

## Tech Stack

- **Backend**: Python Flask with SQLAlchemy
- **Database**: SQLite
- **Machine Learning**: scikit-learn
- **Data Processing**: NumPy, Pandas
- **API Integration**: requests
- **Data Parsing**: BeautifulSoup4

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/weather-forecast.git
cd weather-forecast
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
weathertest/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── templates/         
│   └── index.html     # Frontend template
└── weather.db         # SQLite database
```

## API Reference

The application uses the National Weather Service API (api.weather.gov) to fetch weather data from the KLGA station.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
