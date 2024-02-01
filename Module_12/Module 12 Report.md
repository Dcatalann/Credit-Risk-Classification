Module 12 Report

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The purpose of the Credit Risk Analysis is to provide lenders with how healthy or risky the loan is. This helps them seek patterns and comprehend customer behavior. 

* Explain what financial information the data was on, and what you needed to predict.
The Financial information on the Lending Data provides the summary of the loan, which includes the amount, borrowers income, how much they owe, status and addition loans they may have. 
To predict, I needed to create the X and Y variables from the Dataframe, split the data to Train and Test Dataframes with the train_test_split from  sklearn.model_selection. The data had to be fitted to the logistic_regression_model, and the next step is to predict with the X_test Dataframe. This will give you an array of predictions. 

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
To predict, I needed to create the X and Y variables from the Dataframe, split the data to Train and Test Dataframes with the train_test_split from  sklearn.model_selection. The data had to be fitted to the logistic_regression_model, and the next step is to predict with the X_test Dataframe. This will give you an array of predictions. 


* Describe the stages of the machine learning process you went through as part of this analysis.
Split the Data into Training and Testing Sets, Create a Logistic Regression Model with the Original Data,Predict a Logistic Regression Model with Resampled Training Data

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
The method's that were used to analyze this data were LogisticRegression, RandomOverSampler,confusion matrix, and balanced_accuracy_score. 

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  
        precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
                precision    recall  f1-score   support

           0       0.99      0.99      0.99     56271
           1       0.99      0.99      0.99     56271

    accuracy                           0.99    112542
   macro avg       0.99      0.99      0.99    112542
weighted avg       0.99      0.99      0.99    112542


## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
I believe Module 2 is the one that performs the best. It will give you accurate scores always and is low risk. 

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
In this case, it is better to predict the 1's (higher risk) because it is a lending business and you could lose a high amount of money lending money to someone. The credit risk analysis will be used in a efficient manner to predict high risk borrowers. 


