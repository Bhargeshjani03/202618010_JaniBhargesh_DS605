Assignment Title:-Scikit-learn: Data Preprocessing and Model Performance Evaluation
Name:-Jani Bhargesh
Student_id:-202618010
Dataset Link:-https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
Preprocessing Choices:-knnimputer,StandardScaler and MinmaxNormalisation
Final Observations:-Decision Tree + Standard Scaler Gives the best overall result based on the testing accuracy and F1-Score
2.Scaling has a very small effect on logistic regression.However while comparing both o them standardscaler performs slightly better than minmaxscaler as per the calculate f1-score

3.Scaling has a negligible effect on decision tree as a f1-score is not varied by much after applying scaling

4.Logistic Regression does not show much overfitting as per evidence provided by train-test gap which is very minimal compared to decision tree

5.Decision Tree shows overfitting whihch can be derived from its large train-test gap