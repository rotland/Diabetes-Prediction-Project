# Diabetes Prediction using Support Vector Machine (SVM)

A simple machine learning project that predicts whether a person has diabetes based on medical information such as glucose level, BMI, and age. The model is built in Python using a Support Vector Machine (SVM) classifier from scikit-learn.

## Project Objectives

1. Clean, explore, and prepare the dataset for machine learning.
2. Build an SVM classification model and evaluate it using accuracy.
3. Predict whether a person is diabetic from their medical data.

## Dataset

The project uses the **Pima Indians Diabetes Dataset**, which contains **768 records** and **9 columns**.

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes family history score |
| Age | Age in years |
| **Outcome** | **Target: 1 = diabetic, 0 = not diabetic** |

Class distribution: 500 non-diabetic (0) and 268 diabetic (1).

## Tools and Libraries

- Python 3
- pandas and NumPy (data handling)
- seaborn and matplotlib (visualisation)
- scikit-learn (preprocessing, SVM, evaluation)

## Project Workflow

1. **Import the data** and inspect it (`head`, `shape`, `describe`, class counts).
2. **Check for missing values** (none found).
3. **Separate features (X) and target (y)**.
4. **Standardise the features** using `StandardScaler` so all values are on the same scale.
5. **Split the data** into 80% training and 20% testing sets.
6. **Train an SVM** with a linear kernel (`SVC(kernel='linear')`).
7. **Evaluate** the model on training and test data.
8. **Predict** on a new, unseen patient sample.

## Results

| Dataset | Accuracy |
|---|---|
| Training | ~78% |
| Testing | ~78% |

Because training and test accuracy are very close, the model is not overfitting and generalises reasonably well. Exact numbers may vary slightly between runs because the train/test split is random.

## Example Prediction

```python
input_sample = (5, 166, 72, 19, 175, 22.7, 0.6, 51)
# The sample is standardised with the same scaler, then passed to the model
# Output: person is diabetic
```

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/rotland/diabetes-prediction-svm.git
   cd diabetes-prediction-svm
   ```
2. Install the requirements:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn jupyter
   ```
3. Download the dataset (`diabetes.csv`) and place it in the project folder.
4. Update the file path in the notebook, for example:
   ```python
   Data = pd.read_csv("diabetes.csv")
   ```
5. Open the notebook and run all cells:
   ```bash
   jupyter notebook
   ```

## Possible Improvements

- Handle the zero values in columns like Glucose, BloodPressure, SkinThickness, Insulin, and BMI, which are likely missing data rather than real measurements.
- Fit the scaler on the training data only, then apply it to the test data, to avoid data leakage.
- Add more evaluation metrics such as precision, recall, F1-score, and a confusion matrix, since the classes are imbalanced.
- Try other kernels (RBF, polynomial) and tune hyperparameters with `GridSearchCV`.
- Compare with other models such as Logistic Regression or Random Forest.

## Disclaimer

This project is for learning purposes only. It should not be used as a substitute for professional medical diagnosis.

## Author

rotland – [GitHub](https://github.com/rotland)
