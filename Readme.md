# Predicting House Prices in King County

This repository contains my final assessment project for the Data Analytics with Python course. The objective of this project was to clean, analyze, and build a series of predictive machine learning models using the King County housing dataset (which covers home sales in the Seattle area) to estimate residential property prices based on various features.

Everything is self-contained inside the `house.ipynb` notebook, which walks through the entire data pipeline from initial data preparation to model deployment and evaluation.

## What I Did in This Project

Instead of just jumping straight into machine learning, the project required a thorough exploratory phase to deeply understand the underlying data. Here is the workflow I executed:

* **Digging into the Data:** Cleaned up the dataset, handled missing entries, dropped irrelevant identifiers like `id`, and evaluated the descriptive statistics of the target features.
* **Visualizing the Trends:** Used `seaborn` to generate statistical charts. I looked closely at the premium commanded by waterfront views and analyzed how square footage impacts final sale numbers.
* **Data Prep & Pipelines:** Set up feature scaling using `StandardScaler` and engineered non-linear mappings with `PolynomialFeatures` to help the regression models pick up on trickier, non-linear relationships.
* **Testing Different Models:** I explored a progression of regression architectures—starting from standard linear baseline models and moving up to regularized models—to evaluate performance trade-offs.

## The Results (And What Worked Best)

To figure out how well the models were doing, I used the **$R^2$ score** (the coefficient of determination), which measures what percentage of the price variance the model can explain. Here is how the experiments stacked up across the different requirements:

| Model | Features Used | How I Tested It | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Simple Linear Regression** | Just living room sqft (`sqft_living`) | Full Dataset | **0.493** *(Baseline model)* |
| **Multiple Linear Regression** | 11 key features | Full Dataset | **0.658** |
| **Standardized Pipeline** | 11 features + Scaling | Full Dataset | **0.658** |
| **Ridge Regression ($\alpha=0.1$)** | 11 features | 15% Train/Test Split | **0.648** |
| **Polynomial Ridge Regression** | 11 features (Degree 2) | 15% Train/Test Split | **0.704** |

### Key Takeaway
The best performing model was the **Polynomial Ridge Regression**, which managed to explain roughly **70.4%** of the price variance. Introducing polynomial features allowed the architecture to capture complex interactions between variables (such as how a property's location and its total square footage interact) that a simple straight-line model completely missed.

## Want to Run it Yourself?

If you want to play around with the notebook or check out the plots, here is how to get set up:

### 1. Grab the dependencies
Make sure you have the standard Python data science libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

###2. Fire up the notebook
Download the repository, make sure the kc_house_data.csv dataset is placed in the same folder, and run:
###jupyter notebook's house.ipynb
