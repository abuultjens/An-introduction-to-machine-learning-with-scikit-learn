# An-introduction-to-machine-learning-with-scikit-learn
An introduction to machine learning with scikit-learn


This clinic uses data from this publication:
DOI: 10.1128/aem.01292-23
https://journals.asm.org/doi/10.1128/aem.01292-23

### Setup a conda environment
```
# Make new conda environment
conda create -n ML_bioinfoclinic

# Activate new conda env
conda activate ML_bioinfoclinic

# Install required packages
conda install anaconda::scikit-learn=1.1.2
conda install anaconda::pandas=1.4.2
conda install conda-forge::matplotlib=3.7.1
```

### Clone this GitHub repository:
```
# Clone repo
git clone https://github.com/abuultjens/An_introduction_to_machine_learning_with_scikit-learn.git

# Move into cloned repo
cd An_introduction_to_machine_learning_with_scikit-learn/
```

### Unzip data files:
```
gunzip 421-534_SKA_align_m-0.2_k-15_p-0.1.OHE.csv.gz
gunzip 113-534_SKA_align_m-0.2_k-15_p-0.1.OHE.csv.gz
```

### Start python interpreter and load packages:
```
# Start python
python

# Load packages
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
```

### Load training dataset:
```
# load data
train_data = pd.read_csv('421-534_SKA_align_m-0.2_k-15_p-0.1.OHE.csv', index_col=0)

# Transpose datasets to have observations as rows and features as columns
train_data_t = train_data.transpose()

# Load the train label files
train_labels = pd.read_csv('target_421_NY-2.csv', index_col=0)

# Ensure labels are aligned with data
train_labels = train_labels.loc[train_data_t.index]
```

### Train a ML classifier (choose either a random forest or logistic regression model):

#### Train a random forest classifier:
```
# Load model
from sklearn.ensemble import RandomForestClassifier

# Initialise the Random Forest classifier
classifier_model = RandomForestClassifier()

# Train the classifier with the training data and labels
classifier_model.fit(train_data_t, train_labels.values.ravel())
```

#### Train a logistic regression classifier:
```
# Load model
from sklearn.linear_model import LogisticRegression

# Initialise the Logistic Regression classifier
classifier_model = LogisticRegression()

# Train the classifier with the training data and labels
classifier_model.fit(train_data_t, train_labels.values.ravel())
```

### Load test dataset:
```
# load the test data
test_data_path = '113-534_SKA_align_m-0.2_k-15_p-0.1.OHE.csv'
test_data = pd.read_csv(test_data_path, index_col=0)

# Transpose datasets to have observations as rows and features as columns
test_data_t = test_data.transpose()

# Load the test label files
test_labels_path = 'target_113_NY-2.csv'
test_labels = pd.read_csv(test_labels_path, index_col=0)

# Ensure labels are aligned with data
test_labels = test_labels.loc[test_data_t.index]
```

### Apply the trained model to make classifications on unseen test data:
```
# Make predictions on the test data (unseen during model training)
test_predictions = classifier_model.predict(test_data_t)
```

### Evaluate the model with the test labels:
```
conf_matrix = confusion_matrix(test_labels, test_predictions)
roc_auc = roc_auc_score(test_labels, classifier_model.predict_proba(test_data_t)[:, 1])

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(test_labels, classifier_model.predict_proba(test_data_t)[:, 1])

# Display the confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Display the ROC AUC score
print("ROC AUC: {:.3f}".format(roc_auc))
```

### Make a plot of the ROC curve:
```
# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
```

### Sanity check the analysis (randomise the order of labels):
```
# Load training dataset:
train_data = pd.read_csv('421-534_SKA_align_m-0.2_k-15_p-0.1.OHE.csv', index_col=0)

# Transpose datasets to have observations as rows and features as columns
train_data_t = train_data.transpose()

# Load the train label files
train_labels = pd.read_csv('target_421_NY-2.csv', index_col=0)

#---------------------------------------------
####### randomise the labels #######
train_labels = train_labels.sample(frac=1)
#---------------------------------------------

# Train a random forest classifier:
# Initialise the Random Forest classifier
classifier_model = RandomForestClassifier()

# Train the classifier with the training data and labels
classifier_model.fit(train_data_t, train_labels.values.ravel())

# load the test data
test_data_path = '113-534_SKA_align_m-0.2_k-15_p-0.1.OHE.csv'
test_data = pd.read_csv(test_data_path, index_col=0)

# Transpose datasets to have observations as rows and features as columns
test_data_t = test_data.transpose()

# Load the test label files
test_labels_path = 'target_113_NY-2.csv'
test_labels = pd.read_csv(test_labels_path, index_col=0)

# Ensure labels are aligned with data
#test_labels = test_labels.loc[test_data_t.index]

# Apply the trained model to make classifications on unseen test data:
# Make predictions on the test data (unseen during model training)
test_predictions = classifier_model.predict(test_data_t)

conf_matrix = confusion_matrix(test_labels, test_predictions)
roc_auc = roc_auc_score(test_labels, classifier_model.predict_proba(test_data_t)[:, 1])

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(test_labels, classifier_model.predict_proba(test_data_t)[:, 1])

# Display the confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Display the ROC AUC score
print("ROC AUC: {:.3f}".format(roc_auc))

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

```








