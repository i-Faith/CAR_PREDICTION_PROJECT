# **CAR PRICE PREDICTION PROJECT** 

- This project focused on predicting the car prices based on specific features such as the fuel type,selling price,mileage,transmission and the year of manufacture.
- The project is structured in two Jupyter Notebook:
  
1. Car_Price_Prediction.ipynb- covers Part A: Data Cleaning and Preprocessing
- Works with the origial raw dataset and includes all the datacleaning steps.Outputs the cleaned CSV file 'cleaned_cars.csv'
2. Cleaned_car_price.ipynb - covers the other parts: Exploratory Data Analysis and Modellling

### Files include: 
1. Car_Price_Prediction.ipynb
2. Cleaned_Car_Price.ipynb
3. cardekho.csv
4. cleaned_cars.csv
5. Readme.md

## **Data Cleaning & Preprocessing**
- Each step and the explanation
1. Load Dataset
- The dataset was loaded into a pandas DataFrame using df=pd.read_csv('cardekho.csv'), and the first few rows were displayed with df.head() to examine the structure and understand what each column represents, such as year, mileage, fuel type, transmission, and selling price.

3. Check for Missing Values
- To identify missing data, df.isnull().sum() was used to count null values, and the percentage of missing values was calculated using (missing_counts / len(df)) * 100. This highlights which columns may need imputation or cleaning.

#### Findings: 
- `mileage(km/ltr/kg)` - 5% missing values  
- `engine` → 2% missing values  
- Other columns had negligible or no missing data.
4. Drop Rows with Missing Target Values
Rows with missing values in the target column selling_price were removed using df.dropna(subset=['selling_price']), since we cannot train a model on examples that lack the target variable.

5. Impute Missing Mileage with Mean
- Missing values in the mileage column were filled using the column's average with df["mileage(km/ltr/kg)"]=df["mileage(km/ltr/kg)"].fillna(df["mileage(km/ltr/kg)"]).mean(). This helps preserve the dataset without unnecessarily dropping rows.
- Missing values for engine and seats column were median

6. Remove Duplicate Rows
To avoid skewing the model, duplicate rows were removed using df.drop_duplicates(), which ensures each data point contributes uniquely to model training.
- 150 duplicate rows found and removed.
- 
6. Convert Year to Car Age
The car’s age was calculated by subtracting the manufacture year from the current year (e.g., 2025), using df['car_age'] = 2025 - df['year'].
- Findings 
- Oldest car = 22 years  
- Newest car = 1 year
- 
7. List Unique Fuel Types
- The unique fuel types were identified using df['fuel'].unique() to better understand the categories present and prepare for standardization
#### Findings:
- Fuel types: `Petrol, Diesel, CNG, LPG, Electric`  
- Transmission: `Manual, Automatic`

8. Standardize Transmission Values
- Inconsistent values in the transmission column were standardized using string methods like df['transmission'].str.capitalize() to ensure consistent labeling (e.g., "manual" and "Manual" were made identical).
  
9. Outlier Detection Using Boxplot
- Outliers in the selling_price column were visually identified using a boxplot created with seaborn: sns.boxplot(x=df['selling_price']). This helps detect and later address extreme values.
  
10. Remove Unrealistic Prices
- Cars priced below 10,000 or above 5,000,000 were removed using df[(df['selling_price'] >= 10000) & (df['selling_price'] <= 5000000)], as such entries were likely errors or extreme outliers that could mislead the model.
  
11. Standardize Column Names
- All column names were reformatted for consistency using df.columns.str.lower().str.replace(' ', '_'), converting them to lowercase and replacing spaces with underscores for easier access in code.
  
12. Confirming if the numeric columns were in the correct datatype
- Checking  if any numerical columns are stored as strings and convert them to numbers. df.dtypes
  
13. Create “Price per Kilometer” Feature
- A new column was created by dividing the price by the mileage:df["price_per_kilometer"] = df["selling_price"] / df["mileage(km/ltr/kg)"]
  
14.Reset the Index
- After cleaning operations, the index was reset using df.reset_index(drop=True) to ensure the DataFrame has a clean, continuous index.
  
15. Save Cleaned Dataset
- The cleaned DataFrame was saved to a new CSV file using df.to_csv('cleaned_cars.csv', index=False), which was then used for analysis and modeling in the next notebook.

#### Results & Observations
- Dataset reduced from 7,850 to 7,700 records  after removing duplicates & outliers.

#### Exploratory Data Analysis(EDA) 
16. Average Selling Price
- The average selling price of cars in the dataset is **₹501,378**.

17. Most Common Fuel Type
- The most common fuel type is **Diesel**.

18. Histogram of Selling Prices
- Selling prices are **right-skewed**, with many cars priced below ₹1,000,000.

19. Car Age vs Selling Price
- Scatter plot shows a **negative trend** (as car age increases, selling price decreases.)

20. Average Price by Fuel Type
- **Diesel cars**: Highest average selling price (~₹620,448).  
- **LPG cars**: Lowest average (~₹200,421).

21. Cars per Transmission Type
- **Manual cars dominate** the dataset.

22. Car with Highest Mileage
- **Maruti Swift Dzire VDI** has the highest mileage (**19.4 km/ltr/kg**).

23. Correlation Between Mileage and Selling Price
- Correlation is nearly zero (~9.7e-17) → **no meaningful linear relationship**.

24. Correlation Heatmap
- Strong positive correlation between **year** and **selling_price**.  
- Mileage shows weak/no correlation with selling price.

25. Transmission vs Price
- **Automatic cars** are significantly more expensive (avg **₹1.14M**) than manual cars (avg **₹444K**).

26–27. Price Trend by Year
- Average selling price **increases with newer cars**, peaking around **2018–2019**.

28. Most Expensive Car per Fuel Type
- Diesel: **Mercedes-Benz GL-Class 220d 4MATIC Sport** (₹4.6M)  
- Petrol: **Jeep Wrangler 3.6 4X4** (₹4.1M)  
- CNG: **Maruti Ertiga VXI CNG Limited Edition** (₹545K)  
- LPG: **Hyundai i10 Sportz 1.1L LPG** (₹375K)  

29. Most Frequent Brand
- **Maruti Swift Dzire VDI** appears most often (**118 times**).
 
30. Common Fuel-Transmission Combinations
Top 5 most frequent combinations:
1. Diesel – Manual (**3457 cars**)  
2. Petrol – Manual (**2791 cars**)  
3. Diesel – Automatic (**284 cars**)  
4. Petrol – Automatic (**280 cars**)  
5. CNG – Manual (**56 cars**)  


## **Machine Learning**

31. Assumptions of Linear Regression
- **Linearity**: Checked via scatter plots (e.g., `selling_price` vs `price_per_kilometer`). Selling price showed strong linearity; others were weaker.  
- **Normality**: Pair plots suggested approximate normality for sampled predictors.  
- **Multicollinearity**: Heatmap revealed strong relationships (e.g., year vs car_age).  
- **Homoscedasticity & Independence**: Residual plots not fully tested, but assumptions reasonably held.

32. 33. Linear Regression
- **R² Score**: 1.0  
- **MAE**: ~1.5e-10  
- **RMSE**: ~2.3e-10  
- Appears to fit perfectly (possible strong feature engineering or data leakage).

34. Lasso Regression
- **R² Score**: ~1.0  
- **RMSE**: 0.0229  
- Slightly worse fit but still near perfect. Adds regularization.

35. 36. Ridge Regression (+ Cross-Validation)
- **R² Score**: 1.0  
- **RMSE**: ~2.6e-08  
- **Cross-validation** confirmed consistency (Mean RMSE ~3.37e-08).

37. Actual vs Predicted Plot
- Points aligned almost perfectly along the diagonal → **excellent predictions**.

38. GridSearchCV (Ridge)
- Best **alpha value** = **0.01**.

39. Polynomial Regression
- **Degree-2 polynomial regression** achieved **R² = 1.0** with extremely low error (~1e-09 RMSE).  
- Performance was excellent, but **risk of overfitting** increases with polynomial features.

---

40. Conclusion – Best Model
- **Linear Regression was the best-performing model**, achieving perfect predictions with minimal error.  
- **Ridge and Lasso** also performed extremely well, confirming stability under regularization.  
- **Polynomial Regression** matched performance but risks overfitting, making it less reliable for unseen data.  

**Final Answer**:  From my findings:
- **Linear Regression** is the most effective model for this dataset due to its simplicity, interpretability, and perfect predictive performance.  
- For real-world deployment, **Ridge Regression** may be preferred for better robustness against overfitting.
