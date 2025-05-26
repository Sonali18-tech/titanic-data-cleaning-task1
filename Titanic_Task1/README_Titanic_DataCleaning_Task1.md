# README - Titanic Data Cleaning Task

## Overview
This Python script performs data cleaning and preprocessing on the Titanic dataset to prepare it for machine learning analysis. The script handles missing values, removes irrelevant columns, encodes categorical variables, normalizes numerical features, and visualizes the data.

## Dataset Information
- **Source**: Titanic passenger data (typically includes information like survival status, passenger class, age, fare, etc.)
- **Original Features**: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
- **Final Features After Cleaning**: Survived, Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S

## Data Cleaning Steps

### 1. Loading and Initial Inspection
- The dataset is loaded using pandas
- Basic information is displayed:
  - First few rows (`df.head()`)
  - Dataset shape (`df.shape`)
  - Summary information (`df.info()`)
  - Descriptive statistics (`df.describe()`)
  - Missing value count (`df.isnull().sum()`)

### 2. Handling Missing Values
- **Age**: Missing values are filled with the median age
- **Embarked**: Missing values are filled with the most common embarkation port (mode)
- **Cabin**: This column is dropped as it has too many missing values

### 3. Removing Irrelevant Columns
The following columns are removed as they are not useful for analysis:
- `PassengerId` (unique identifier)
- `Name` (individual names)
- `Ticket` (ticket numbers)
- `Cabin` (too many missing values)

### 4. Encoding Categorical Variables
- `Sex` and `Embarked` columns are converted to numerical values using one-hot encoding
- For `Sex`: Becomes `Sex_male` (1 for male, 0 for female)
- For `Embarked`: Becomes `Embarked_Q` and `Embarked_S` (with `C` as reference)

### 5. Normalizing Numerical Features
- `Age` and `Fare` are standardized using `StandardScaler`
- This transforms them to have mean=0 and standard deviation=1

### 6. Outlier Detection and Removal
- A boxplot is created to visualize outliers in the `Fare` column
- Extreme outliers are removed (values with z-score > 3)

### 7. Data Visualization
- Survival count plot (`sns.countplot`)
- Correlation heatmap (`sns.heatmap`)

### 8. Saving Cleaned Data
- The cleaned dataset is saved as `cleaned_titanic.csv`

## How to Use This Script

### Prerequisites
- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn

### Installation
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

### Running the Script
1. Place the script in the same directory as your `Titanic-Dataset.csv` file
2. Update the file path in the script if needed:
   ```python
   df = pd.read_csv(r"C:\Users\capl2\OneDrive\Pictures\Documents\Titanic_Task1\Titanic-Dataset.csv")
   ```
3. Run the script:
   ```bash
   python task1_titanic_cleaning.py
   ```

### Output
- The script will:
  - Display various information about the dataset in your console
  - Show three plots (boxplot, countplot, heatmap)
  - Save the cleaned dataset as `cleaned_titanic.csv`

## Notes 
1. **Missing Values**: We handle them by either filling with median/mode or removing columns with too many missing values
2. **Categorical Encoding**: Converting text categories to numbers that algorithms can understand
3. **Normalization**: Scaling numerical features to similar ranges helps many machine learning algorithms perform better
4. **Outliers**: Extreme values that can distort analysis - we visualize and remove the most extreme cases
5. **Visualization**: Helps understand the data distribution and relationships between variables

## Next Steps
This cleaned dataset is now ready for:
- Exploratory data analysis
- Feature engineering
- Machine learning model training

## Troubleshooting
- If you get file not found errors, check the file path is correct
- If you get module not found errors, ensure you've installed all required packages
- The outlier removal threshold (3) can be adjusted based on your specific dataset