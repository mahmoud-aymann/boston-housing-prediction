# Boston Housing Price Prediction

A machine learning project that predicts Boston housing prices using a Decision Tree Regressor with data normalization. The project includes a complete Jupyter notebook analysis and a Flask web application for interactive predictions.

## Project Structure

```
boston-housing/
├── boston_housing.ipynb      # Complete Jupyter notebook analysis
├── housing.csv               # Boston housing dataset
├── train_model.py           # Model training script with normalization
├── app.py                   # Flask web application
├── templates/
│   └── index.html          # Web interface template
├── boston_housing_model.pkl # Trained model with normalization (generated)
├── model_info.pkl          # Model metadata (generated)
└── requirements.txt        # Python dependencies
```

## Features

- **Complete Data Analysis**: Exploratory data analysis with statistics and visualizations
- **Data Normalization**: PowerTransformer normalization for better model performance
- **Machine Learning Model**: Decision Tree Regressor with optimal hyperparameters
- **Web Application**: Interactive Flask app for making predictions
- **Model Persistence**: Pickle files for easy model deployment
- **Responsive UI**: Modern, mobile-friendly web interface

## Dataset

The dataset contains 489 samples with 3 features:
- **RM**: Average number of rooms per dwelling
- **LSTAT**: Percentage of lower status population
- **PTRATIO**: Pupil-teacher ratio by town

Target variable: **MEDV** (Median value of owner-occupied homes in $1000's)

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if not already done):
   ```bash
   python train_model.py
   ```

## Usage

### 1. Jupyter Notebook Analysis
Open `boston_housing.ipynb` to see the complete analysis including:
- Data exploration and statistics
- Feature analysis and insights
- Model training and evaluation
- Learning curves and complexity analysis
- Answers to all theoretical questions

### 2. Flask Web Application
Start the web application:
```bash
python app.py
```

Then open your browser and go to: `http://localhost:5000`

The web app allows you to:
- Input house features (rooms, poverty level, student-teacher ratio)
- Get instant price predictions
- View sample predictions
- Learn about feature meanings

### 3. API Usage
The Flask app also provides REST API endpoints:

- `GET /api/info` - Get model information
- `POST /predict` - Make predictions
- `GET /api/sample_predictions` - Get sample predictions

Example API call:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"rm": 6, "lstat": 15, "ptratio": 15}'
```

## Model Performance

- **Algorithm**: Decision Tree Regressor with PowerTransformer
- **Normalization**: PowerTransformer (yeo-johnson method)
- **Optimal Depth**: 5
- **R² Score**: 0.7879
- **Cross-validation**: 10-fold with ShuffleSplit

## Sample Predictions

| Rooms | Poverty % | Student-Teacher Ratio | Predicted Price |
|-------|-----------|----------------------|-----------------|
| 5     | 17        | 15                   | $419,700        |
| 4     | 32        | 22                   | $287,100        |
| 8     | 3         | 12                   | $927,500        |

## Key Insights

1. **Room Count (RM)**: More rooms generally mean higher prices
2. **Poverty Level (LSTAT)**: Lower poverty levels correlate with higher property values
3. **Student-Teacher Ratio (PTRATIO)**: Better schools (lower ratios) increase property values

## Technical Details

- **Python Version**: 3.8+
- **Key Libraries**: scikit-learn, pandas, numpy, Flask
- **Model Type**: Decision Tree Regressor
- **Validation**: Grid Search with Cross-Validation
- **Deployment**: Flask web application with pickle serialization

## Files Generated

After running `train_model.py`, the following files are created:
- `boston_housing_model.pkl`: The trained model
- `model_info.pkl`: Model metadata and feature information

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## License

This project is for educational purposes and follows standard machine learning practices.

