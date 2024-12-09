## Full name and email address:
Lee Jia Suen 
jslee009@suss.edu.sg 


## A. Key Findings from EDA and Preprocessing 

1. *Nutrient P Sensor (ppm)*, *Nutrient K Sensor (ppm)* and *Nutrient N Sensor (ppm)* has string values. The numerical values ends with a "ppm"  that needs to be removed. These columns were converted to float64 data after.
   
2. Duplicated rows of 7489 were dropped.
   
3. Missing values from Temperature Sensor (°C) , Humidity Sensor (%), Light Intensity Sensor (lux), Nutrient N Sensor (ppm), Nutrient P Sensor (ppm), Nutrient K Sensor (ppm), and Water Level Sensor (mm) were replaced with the median of the columns group by each Plant type-Stage. This is because some plants may have different sensor data hence we preserve the granularities of each Plant type-Stage. From the bar charts we could see the proportion of missing data to each Plant type-Stage data is small as well. 

### Categorical Features

4. *Plant Type* and *Plant Stage* has same label but different naming conventions e.g. vine crops and  VINE CROPS. All unique values are identified and treated to have a standardised naming convention.
   
5. The samples of all plant type-stage are balanced, hence we could look at test accuracy instead of just F1 score.
   
6. Label Encoding converts labels into integer values since Models like Random Forest, XGBoost, and Neural Networks can't directly interpret string labels such as 'Fruiting Vegetables-Maturity'.
   
### Numerical Features

7. *Temperature Sensor (°C)* and *Light Intensity Sensor (lux)* has extreme outliers. There is a gap between the lowest negative value and the positive value, which suggests it might be a data entry error. It is also impossible for both sensors to be negative as it does not make sense in context of plants. Since negative values for Temperature Sensor (°C) were just 1065 out of 50,000, we replaced it with the median grouped by Plant type-Stage. Same goes for Light Intensity Sensor (lux).
   
8. The correlation of Temperature Sensor (°C) with the other columns are non-linear as it has little to no correlation abs value, but from the scatterplot graphs we could observe obvious patterns.
   
9. The correlation of Temperature Sensor (°C) with the other columns are non-linear as it has little to no correlation abs value, but from the scatterplot graphs we could observe obvious patterns.
   
10. The average temperature sensor of each Plant type-Stage range from 21°C to 24°C, which is where most data comes from. But this also means it is the optimal temperature for plant growth and the underrepresented temperatures may not be operationally significant hence we do not need to preprocess this. 

11. Standard Scaler is used to normalise the large range values from numerical columns. This prevents large values from dominating the data for Neural Networks. We do this after train test split to prevent information leakage.

## B. Modelling

For all the models, I used 5 fold cross-validation and GridSearchCV to find the best parameters and ensures training is robust. 5 fold cross-validation splits the training data into 5 folds to such that there is generalisation. GridSearchCV ensures the best hyperparameters that yields the lowest MSE.

For predicting continuous variable, Temperature Sensor (°C), we can use regression models. 
Since the numerical features is non-linear to Temperature Sensor (°C), Random Forest (RF), XGBoost, and Neural Networks (NN) are all suitable models to use. These 3 models are also suitable for predicting the categorical target Plant Type-Stage.

1. Random Forest

Can capture non-linear patterns due to its ensemble of decision trees and splits data in a non-linear manner.

2. XGBoost

For complex and non-linear data, it is a boosting algorithm that builds trees sequentially, improving from the errors of previous trees, which is preferred when accuracy is critical.

3. Neural Networks

Can capture full complexity of the non-linear data. The hidden layers use a non-linear activation function and can model a function like sine wave, or exponential curve which is perfect for our data.


## C.Evaluation of the Models 

### Numerical Features:

1. Random Forest
   
- Best Hyperparameters: {'max_depth': 5, 'max_leaf_nodes': 14, 'min_samples_leaf': 300, 'min_samples_split': 300}
- Best Cross-Validation Mean Squared Error (MSE): 0.8963
- Best Cross-Validation Root Mean Squared Error (RMSE): 0.9467
- Test Mean Squared Error (MSE): 0.8706
- Test Root Mean Squared Error (RMSE): 0.9330
- Test R² Score: 0.6458

2. XGBoost
   
- Best Hyperparameters: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'min_child_weight': 5, 'n_estimators': 100, 'subsample': 1.0}
- Best Cross-Validation Mean Squared Error (MSE): 0.7487
- Best Cross-Validation Root Mean Squared Error (RMSE): 0.8653
- Test Mean Squared Error (MSE): 0.7495
- Test Root Mean Squared Error (RMSE): 0.8657
- Test R² Score: 0.6950

3. Neural Network
   
- Neural Network Test MSE: 0.7849
- Neural Network Test RMSE: 0.8859
- Neural Network Test R² Score: 0.6806

Random Forest has the lowest R² Score (0.6458) compared to the other two models, indicating it explains only ~64.6% of the variance in the temperature prediction, hence it struggles to capturing the fine-grained non-linear relationships in the data. RMSE is higher than other two models, which means it has the highest prediction error. 

Neural Network performs slightly worse than XGBoost but better than Random Forest.

XGBoost is the champion model in terms of a lowest RMSE and highest R² Score. The lower RMSE indicates it is better at minimizing the error between predicted and actual values compared to Random Forest.

### Categorical Features:

1. Random Forest

- Best Hyperparameters: {'max_depth': 5, 'max_leaf_nodes': 13, 'min_samples_leaf': 300, 'min_samples_split': 1000}
- Best Cross-Validation Accuracy: 0.9998
- Test Accuracy: 0.9997
- Macro F1: 1.00
  
2. XGBoost

- Best Hyperparameters: {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}
- XGBoost Test Accuracy: 0.9999
- Macro F1: 1.00

3. Neural Network 
- Test Accuracy: 1.0000
- Macro F1: 1.00

Random Forest, XGBoost, and Neural Network all achieved test accuracies near or equal to 1.00, which shows that the models have successfully learned the patterns in the data.

For Random Forest and XGBoost, the cross-validation accuracy is also very high and very close to the test accuracy, which showed excellent generalizability.

The Neural Network achieved a perfect test accuracy of 1.00 and Macro F1, which means it successfully learned the underlying patterns in the sensor data and their relationship with Plant Type-Stage. 


