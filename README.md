# House Price Prediction

A machine learning project that predicts house prices from structural and locational features using the classic Housing Prices dataset. The project walks through data exploration, cleaning, model building, and visualization in a single, well-organized Jupyter notebook.

## Overview

This project compares **Linear Regression** and **Random Forest Regression** models to predict house prices based on 12 features including area, number of bedrooms/bathrooms, number of stories, and amenities like air conditioning and a preferred location flag.

The full workflow is documented in [`analysis.ipynb`](./analysis.ipynb) and is organized into five clear tasks:

1. **Data Loading & Exploration** — load the dataset, inspect shape, types, and summary statistics
2. **Data Cleaning** — check for nulls/duplicates and one-hot encode categorical features
3. **Model Building** — train and evaluate Linear Regression and Random Forest models
4. **Visualization** — generate charts for price distribution, feature correlation, and prediction accuracy
5. **Insights & Summary** — key takeaways from the analysis

## Dataset

The dataset (`Housing.csv`) contains **545 records** with **13 columns**:

| Column | Description |
|---|---|
| `price` | Sale price of the house (target variable) |
| `area` | Area of the plot in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories |
| `mainroad` | Whether the house faces a main road (yes/no) |
| `guestroom` | Whether the house has a guest room (yes/no) |
| `basement` | Whether the house has a basement (yes/no) |
| `hotwaterheating` | Whether the house has hot water heating (yes/no) |
| `airconditioning` | Whether the house has air conditioning (yes/no) |
| `parking` | Number of parking spots |
| `prefarea` | Whether located in a preferred area (yes/no) |
| `furnishingstatus` | Furnishing status (furnished / semi-furnished / unfurnished) |

The dataset has no missing values and no duplicate rows.

## Methodology

1. **Exploration** — inspected dataset shape, column types, and descriptive statistics (`df.describe()`, `df.info()`).
2. **Cleaning** — removed duplicate rows and one-hot encoded all categorical columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`) using `pd.get_dummies()` with `drop_first=True`, expanding the dataset to 14 columns.
3. **Modeling** — split the data 80/20 into train and test sets (`random_state=42`) and trained two models:
   - **Linear Regression** (`sklearn.linear_model.LinearRegression`)
   - **Random Forest Regression** (`sklearn.ensemble.RandomForestRegressor`, 100 estimators)
4. **Evaluation** — compared models using MAE, RMSE, and R².
5. **Visualization** — plotted price distribution, a feature correlation heatmap, and actual vs. predicted prices for the Random Forest model. Charts are saved to the [`charts/`](./charts) folder.

## Results

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 970,043 | 1,324,507 | 0.653 |
| Random Forest | 1,021,546 | 1,400,566 | 0.612 |

Linear Regression slightly outperformed Random Forest on this dataset, achieving a lower error and a higher R² score. Area, number of bathrooms, and furnishing status showed the strongest correlation with price.

## Project Structure

```
house-price-prediction/
├── analysis.ipynb     # Main notebook: exploration, cleaning, modeling, visualization
├── Housing.csv         # Dataset (545 rows × 13 columns)
├── charts/              # Generated visualizations
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   └── actual_vs_predicted.png
├── summary.docx         # Project summary document
├── LICENSE
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/dewanshikarnawat/house-price-prediction.git
   cd house-price-prediction
   ```

2. Install the required dependencies
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. Launch the notebook
   ```bash
   jupyter notebook analysis.ipynb
   ```

4. Run all cells to reproduce the data exploration, model training, and visualizations.

## Tech Stack

- **Python**
- **pandas** & **numpy** — data manipulation
- **scikit-learn** — model training and evaluation
- **matplotlib** & **seaborn** — data visualization

## License

This project is licensed under the terms of the [MIT License](./LICENSE).

## Author

**Dewanshi Karnawat**
GitHub: [@dewanshikarnawat](https://github.com/dewanshikarnawat)
