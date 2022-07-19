# BMI_Prediction_of_Health_Status
Personal Project: Comparison of various machine learning models to determine the algorithm that best predicts BMI classification.

To approach this unbalanced classfication problem, I trained and evaluated models with unbalanced classes. I identified 5 BMI indeces from the dataset: # Index : 0 - Extremely Weak 1 - Weak 2 - Normal 3 - Overweight 4 - Obesity 5 - Extreme Obesity. Instead of having 5 categorizations, I converted the target column into two: "normal" and "at risk". I chose to do this for ease of finding an efficient model that suits the data, but can also be used when new data is introduced.

I oversampled the data using the RandomOverSampler and SMOTE algorithms, and undersampled the data using the RandomUnderSampler algorithm. Then, I used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. 

I then compared two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict efficacy of BMI classification.

Lastly, I evaluated the performance of these models to make an effective comparison and recommendation towards the best model. 

This is an excellent use of machine learning that can be applied in medical research and by health professionals.


## Tools:

Jupyter Notebook

## Language:

Python

## Libraries:

Pandas, Numpy, Imbalanced-learn, Scikit-learn, Collections, Warnings 

## Dataset:

https://www.kaggle.com/code/devprabal/bmi-prediction-of-health-status/data
