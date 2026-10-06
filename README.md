# Distribution Transformer Alarm Classification Using Machine Learning

## What the notebook does

The notebook classifies the magnetic oil-gauge alarm state (`MOG_A`) using 16 temperature, oil-level, voltage and current features. It identifies the recorded alarm state from measurements at the same timestamp.

The workflow combines `Overview.csv` and `CurrentVoltage.csv` by timestamp, explores the measurements and alarm counts, and compares eight machine-learning classifiers. It uses an 80/20 train–test split with `random_state=21` and MinMax scaling fitted on the training data. Evaluation includes accuracy, class-wise precision, recall, F1-score and confusion matrices.

KNN uses three neighbors. Random Forest and Extra Trees each use 100 estimators; the remaining model parameters use library defaults. The implementation uses Python, scikit-learn, XGBoost, pandas, NumPy, Matplotlib and Seaborn.

## Results

The following accuracies are recorded in the notebook's saved run.

| Model | Training accuracy | Test accuracy |
| --- | ---: | ---: |
| **Extra Trees** | **100.00%** | **99.02%** |
| Random Forest | 100.00% | 99.00% |
| XGBoost | 100.00% | 98.80% |
| Decision Tree | 100.00% | 98.56% |
| AdaBoost | 98.20% | 97.83% |
| Support Vector Machine | 95.41% | 95.55% |
| K-Nearest Neighbors | 98.08% | 95.48% |
| Logistic Regression | 94.22% | 94.28% |

**Extra Trees achieved the highest test accuracy among the eight models.** Its alarm-class precision was **0.94**, recall **0.96**, and F1-score **0.95**. These results describe classification accuracy; ROC-AUC was not calculated.

## Conclusion

Tree ensembles can capture nonlinear thresholds and interactions between sensor measurements, which may explain their stronger performance in this comparison. Extra Trees and Random Forest performed almost identically: their 0.02 percentage-point accuracy difference does not establish a consistent advantage.

The random split can place nearby or repeated measurements in both training and test sets, so these scores may be optimistic for future observations. The results support alarm-state classification within this dataset; performance on later periods or different transformers remains untested.
