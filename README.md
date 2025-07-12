# Salary Prediction Using Adult Income Dataset

This project uses the **Adult Income Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult) to build a model that predicts whether a person earns more than \$50K per year based on demographic and economic features.

## 📁 Files

- `Salary-Prediction.ipynb`: Jupyter notebook containing all the code and analysis for this project.
- `Code-Report.pdf`: Project overview and explanation.

## 📊 Dataset

The dataset includes the following:
- A training set (`adult.data`)
- A test set (`adult.test`)

### Features include:
- Age
- Workclass
- Education and Education-Num
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain / Loss
- Hours per Week
- Native Country
- Income (target variable)

## 🧪 Project Steps

1. **Load Libraries**: Import essential Python packages like `pandas`, `numpy`, `matplotlib`, and `seaborn`.
2. **Data Loading**: Download and load both training and test sets from the UCI repository.
3. **Data Cleaning**:
   - Handle missing values represented as `'?'`
   - Clean test set labels (e.g., removing trailing `.` in income classes)
4. **Exploratory Data Analysis (EDA)**:
   - Visualize feature distributions
   - Understand correlations and feature-target relationships
5. **Data Preprocessing**:
   - Encode categorical variables
   - Normalize/standardize features if needed
6. **Modeling**:
   - Train models like Logistic Regression, Decision Trees, KNN, XGBoost, SVM, Random Forests, and MLP
   - Evaluate models using metrics such as accuracy, precision, recall, and F1-score
7. **Hyperparameter Tuning**:
   - Use **GridSearchCV** or **RandomizedSearchCV** to optimize model parameters
   - Improve model generalization and performance
   - Example tuned parameters:
     - `max_depth`, `min_samples_split`, `n_estimators` for tree-based models
     - `C`, `penalty` for logistic regression
   - Use **cross-validation** to prevent overfitting
   - 
8. **Model Comparison**:  
   - Compare the performance of different algorithms.  
   - Select the best model based on balanced metrics.  

    <img width="534" height="642" alt="image" src="https://github.com/user-attachments/assets/9d0319b3-b92f-4ac5-b214-37d02f5644de" />


    <img width="2990" height="1790" alt="image" src="https://github.com/user-attachments/assets/7aa67357-1b7f-48e3-b734-1db914d2ddcb" />


8. **Conclusion**:
   - Summarize key findings and best-performing model
  
   <img width="805" height="295" alt="image" src="https://github.com/user-attachments/assets/d55854c5-044d-4e9c-88ce-748f6d5eeb4e" />

   



## 📈 Sample Visuals

Visualizations include:

- Distribution of salaries
  
  <img width="1389" height="590" alt="image" src="https://github.com/user-attachments/assets/33e7d62a-fd7b-4bf3-ad21-8a02b346a475" />

- Relationship between age, education, hours worked, and income
  
  <img width="520" height="432" alt="image" src="https://github.com/user-attachments/assets/46c0c4aa-c6a5-48d9-8969-1c1381b2a105" />

- Heatmaps and bar plots for categorical variables
