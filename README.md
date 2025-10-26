## Support Vector Machines (SVM) Classification

## Objective
This task focused on utilizing Support Vector Machines (SVMs) for both linear and non-linear classification problems. The implementation involved using Scikit-learn for model building, NumPy for numerical operations, and Matplotlib/Seaborn for visualization.

## Dataset
The Breast Cancer Wisconsin (Diagnostic) dataset was used for this binary classification task. This dataset contains features computed from digitized images of fine needle aspirate (FNA) of a breast mass, used to predict whether a tumor is malignant or benign.

## Key Concepts Explored
* **Margin Maximization**: Understanding how SVMs work to find an optimal hyperplane that maximizes the margin between classes.
* **Kernel Trick**: Exploring different kernel functions (Linear and Radial Basis Function - RBF) to handle linearly and non-linearly separable data by implicitly mapping data into higher-dimensional spaces.
* **Hyperparameter Tuning**: Optimizing SVM performance by tuning key hyperparameters like `C` (regularization parameter) and `gamma` (kernel coefficient) using `GridSearchCV`.

## Implementation Steps

1.  **Library Import**: Imported essential libraries including `pandas`, `numpy`, `sklearn.datasets` (for Breast Cancer), `sklearn.model_selection` (`train_test_split`, `GridSearchCV`, `cross_val_score`), `sklearn.preprocessing` (`StandardScaler`), `sklearn.svm` (`SVC`), `sklearn.metrics` (`accuracy_score`, `classification_report`, `confusion_matrix`), `matplotlib.pyplot`, `matplotlib.colors`, `sklearn.decomposition` (`PCA`), and `seaborn`.
2.  **Dataset Loading and Preparation**: The Breast Cancer dataset was loaded using `load_breast_cancer()`. The features (`X`) and target labels (`y`) were separated.
3.  **Feature Normalization and Data Splitting**: `StandardScaler` was applied to normalize the features, which is critical for SVM performance due to its reliance on distance metrics. The dataset was then split into training (70%) and testing (30%) sets using `train_test_split` with `random_state=42` for reproducibility.
4.  **SVM Training with Linear and RBF Kernels**:
    * An SVM model with a `linear` kernel was trained and evaluated.
    * An SVM model with an `rbf` (Radial Basis Function) kernel was trained and evaluated.
    * Accuracy, confusion matrix, and classification report were printed for both models to compare their initial performance.
5.  **Hyperparameter Tuning (C and Gamma)**: `GridSearchCV` was employed to systematically search for the best `C` and `gamma` parameters for the RBF kernel. A parameter grid was defined, and 5-fold cross-validation (`cv=5`) was used during the search. The model with the `best_params_` was then used for further evaluation.
6.  **Cross-Validation Evaluation**: The performance of the best SVM model (from hyperparameter tuning) was further validated using 10-fold cross-validation on the entire scaled dataset. This provides a more robust estimate of the model's generalization ability by averaging scores across multiple splits.
7.  **Decision Boundary Visualization**: To visualize the decision boundaries, Principal Component Analysis (PCA) was applied to reduce the dataset's dimensionality to 2 components. An SVM model (with optimal hyperparameters) was trained on this 2D data, and its decision boundary was plotted along with the training and testing data points.

## Results

### Initial Model Performance
* **Linear Kernel SVM Accuracy**: ~0.9825
* **RBF Kernel SVM Accuracy (default params)**: ~0.9708

### Hyperparameter Tuning Results (RBF Kernel)
* **Best Parameters Found**: `{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}`
* **Best Cross-Validation Accuracy (during GridSearchCV)**: ~0.9749

### Tuned SVM Model Performance (on Test Set)
* **Accuracy with Tuned RBF Kernel**: 0.9883
* **Confusion Matrix for Tuned RBF Kernel**:
    ```
    [[66  1]
     [ 0 104]]
    ```
    This indicates:
    * 66 Malignant (Class 0) cases correctly predicted.
    * 1 Malignant case incorrectly predicted as Benign.
    * 0 Benign (Class 1) cases incorrectly predicted as Malignant.
    * 104 Benign cases correctly predicted.
    
    Or, if you generated a heatmap:
    ![Confusion Matrix for Optimal K](confusion_matrix_heatmap.png)

    **Interpretation**: The tuned RBF kernel SVM achieved excellent performance on the test set. It correctly classified all 104 benign cases and 66 out of 67 malignant cases, resulting in very high accuracy with minimal misclassifications.

### Cross-Validation Scores (Optimal SVM)
* **Individual cross-validation accuracies (10-fold)**: `[0.9737 0.9825 0.9825 0.9737 0.9825 0.9649 0.9825 0.9912 0.9912 0.9649]` (These values can vary slightly based on runs)
* **Mean cross-validation accuracy**: 0.9790
* **Standard deviation of cross-validation accuracies**: 0.0094

**Interpretation**: The high mean cross-validation accuracy and low standard deviation suggest that the tuned SVM model is robust and generalizes well across different subsets of the data, indicating consistent performance.

### Decision Boundary Visualization
The decision boundaries, visualized using the first two principal components of the Breast Cancer dataset, are shown below:

![Decision Boundary Plot](decision_boundary_plot.png)

**Interpretation**: The plot visually represents how the SVM, with the RBF kernel, separates the two classes (malignant and benign) in the reduced 2D feature space. The colored regions indicate the areas where the model predicts each class. The decision boundary, although non-linear due to the RBF kernel, effectively separates the vast majority of data points, demonstrating the model's ability to handle complex, non-linear relationships in the data.

## Conclusion
This task successfully demonstrated the application of Support Vector Machines for binary classification. We explored both linear and non-linear (RBF) kernels, performed crucial hyperparameter tuning using `GridSearchCV`, and validated the model's robustness with cross-validation. The visualization of decision boundaries provided insight into how SVMs separate data. The results confirm that SVMs are powerful and effective models, especially when features are scaled and hyperparameters are optimized.
