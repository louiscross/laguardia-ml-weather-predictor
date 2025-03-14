# Weather Forecast Application

A Flask-based weather application that fetches real-time weather data from the National Weather Service API, stores historical data, and provides weather predictions using machine learning.

## Features

- Real-time weather data from NOAA Weather API
- Historical weather data tracking
- Machine learning-based weather predictions for the remaining hours of the day
- Interactive data visualization
- Automatic data updates every 5 minutes

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
