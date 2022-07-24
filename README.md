# BMI_Prediction_of_Health_Status
Personal Project: Comparison of various machine learning models to determine the algorithm that best predicts BMI classification.

To approach this unbalanced classfication problem, I trained and evaluated models with unbalanced classes. I identified 5 BMI indeces from the dataset: # Index : 0 - Extremely Weak 1 - Weak 2 - Normal 3 - Overweight 4 - Obesity 5 - Extreme Obesity. Instead of having 5 categories, I converted the target column into two: "normal" and "at risk". I chose to do this for ease of finding an efficient model that suits the data, but can also be used when new data is introduced.

I oversampled the data using the RandomOverSampler and SMOTE algorithms, and undersampled the data using the RandomUnderSampler algorithm. Then, I used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. 

I then compared two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict efficacy of BMI classification.

Lastly, I evaluated the performance of these models to make an effective comparison and recommendation towards the best model. 

This is an excellent use of machine learning that can be applied in medical research and by health professionals.


### Tools:

Jupyter Notebook

### Language:

Python

### Libraries:

pandas, numpy, imbalanced-learn, scikit-learn, collections, warnings 

### Dataset:

https://www.kaggle.com/code/devprabal/bmi-prediction-of-health-status/data

## Visual Reresentation of Data

https://public.tableau.com/views/BMIIndexCategoryAnalysis/BreakdownofBMIIndexCategoryDataset?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link

<img src="https://github.com/katmarcin/BMI_Prediction_of_Health_Status/blob/44ff67a3eb22b7302e5676adc866d5774d79afab/images/gender_breakdown.jpg" width="680" height="440" />


## Results:

Six machine learning models were evaluated for this analysis: RandomOverSampler, SMOTE algorithm, RandomUnderSampler algorithm, SMOTEENN algorithm, BalancedRandomForestClassifier, and EasyEnsemble Classifier. Images of the outputs for each model are used to supplement this analysis. Outputs for each model were calculated from the imbalanced classification report. This method was imported from imblearnmetrics.


* ## RandomOverSampler:

  * Balanced Accuracy Score: 0.74
  * Precision: The average precision score is 0.87. Separately, the at-risk group had a precision of 0.85 and normal group had a precision of 0.94.
  * F1 Score: 0.85

<img src="https://github.com/katmarcin/BMI_Prediction_of_Health_Status/blob/9d59670d0c67f9dea2b9728641785340f95fe3a0/images/naive.jpg" width="650" height="250" />  
      

* ## SMOTE algorithm:

 
  * Balanced Accuracy Score: 0.74
  * Precision: The average precision score is 0.87. Separately, the at-risk group had a precision of 0.85 and normal group had a precision of 0.94.
  * F1 Score: 0.85
  

<img src="https://github.com/katmarcin/BMI_Prediction_of_Health_Status/blob/9d59670d0c67f9dea2b9728641785340f95fe3a0/images/smote.jpg" width="650" height="250" />  
 
  
* ## RandomUnderSampler algorithm:

 
  * Balanced Accuracy Score: 0.75
  * Precision: The average precision score is 0.87. Separately, the at-risk group had a precision of 0.84 and normal group had a precision of 1.00.
  * F1 Score: 0.85

<img src="https://github.com/katmarcin/BMI_Prediction_of_Health_Status/blob/9d59670d0c67f9dea2b9728641785340f95fe3a0/images/under.jpg" width="650" height="250" />  


* ## SMOTEENN algorithm:

 
  * Balanced Accuracy Score: 0.75
  * Precision: The average precision score is 0.87. Separately, the at-risk group had a precision of 0.84 and normal group had a precision of 1.00.
  * F1 Score: 0.85


<img src="https://github.com/katmarcin/BMI_Prediction_of_Health_Status/blob/9d59670d0c67f9dea2b9728641785340f95fe3a0/images/combo.jpg" width="650" height="250" />     


* ## BalancedRandomForestClassifier:

 
  * Balanced Accuracy Score: 0.87
  * Precision: The average precision score is 0.91. Separately, the at-risk group had a precision of 0.98 and normal group had a precision of 0.48.
  * F1 Score: 0.87

<img src="https://github.com/katmarcin/BMI_Prediction_of_Health_Status/blob/9d59670d0c67f9dea2b9728641785340f95fe3a0/images/random_forest.jpg" width="650" height="350" /> 

* ## EasyEnsembleClassifier:


  * Balanced Accuracy Score: 0.70
  * Precision: The average precision score is 0.83. Separately, the at-risk group had a precision of 0.81 and normal group had a precision of 0.88.
  * F1 Score: 0.79
 
<img src="https://github.com/katmarcin/BMI_Prediction_of_Health_Status/blob/9d59670d0c67f9dea2b9728641785340f95fe3a0/images/ee.jpg" width="650" height="250" /> 

# Summary

The three metrics used to described each machine learning model are balanced accuracy score, precision, and F1 score. Balanced accuracy score is defined as the average of recall obtained on each class. The best balanced accuracy value is 1 and the worst value is 0 when adjusted=False. Of all the six machine learning models produced, EasyEnsembleClassifier had the lowest balanced accuracy value of 0.70. The highest score is seen in the BalancedRandomForestClassifier model with a value of 0.87. 

Next, precision of each model is calculated by the ratio of true positives to the sum of true and false positives in numerical format. All of the resampling models demonstrated the same average precision score of 0.87. However, the most important aspect of the precision is that which reflects the at-risk and normal groups individually. All the normal groups had a higher precision than the majority target feature, at-risk, except for BalancedRandomForestClassifier. In this model, a low precision of 0.48 is demonstrated for the normal group whereas the at-risk group had a precision value of 0.98. Despite this, the best overall average precision score is still seen in the BalancedRandomForestClassifier. However, the difference between the highest and lowest average precision scores only differed by 0.08.

 Lastly, F1 score measures the ability of a classifier to find all positive instances. In other words, it is the fraction of positives that were correctly identified. A good classifier should have a score of 1, and a poor classifier would have a score closer to 0. BalancedRandomForestClassifier shines through again with a high recall score of 0.87. The lowest F1 score, 0.79, is exhibited by the EasyEnsembleClassifier mdoel.

For further BMI prediction analysis, it would be best to proceed with the BalancedRandomForestClassifier, which exhibited relatively high scores amongst all major metrics and therefore it performed the best. It is important to choose this model as it implements an algorithm that reduces bias and is also both precise and accurate in calculating generalized health status risk. Hospitals must have proper health status management and modeling to prevent false negatives, aka at-risk individuals that are misinterpreted as maintaining a normal BMI. In doing so, this leads to healthcare providers believing the patient is not at more risk relative to others because of false pretenses of the BMI. 


