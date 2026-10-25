# DS605 Lab Assignment 8: Decision Trees and Support Vector Machines

## Overview

This lab explores two supervised machine learning algorithms: **Decision Trees** and **Support Vector Machines (SVMs)**. The algorithms are implemented from scratch using NumPy and compared with their scikit-learn implementations.

The lab also investigates how SVM kernels can model nonlinear decision boundaries using the `make_moons` dataset.

## Objectives

- Implement a Decision Tree classifier from scratch using the Gini impurity criterion.
- Implement entropy and information gain to understand alternative splitting criteria.
- Compare scratch Decision Trees with scikit-learn implementations.
- Visualize decision boundaries using two features.
- Implement a linear SVM from scratch using hinge loss and subgradient descent.
- Understand the role of feature standardization, margin maximization, and the regularization parameter \(C\).
- Explore linear, polynomial, and RBF kernels for nonlinear classification.
- Compare models using classification metrics and computational timings.

## Datasets

### 1. Banknote Authentication Dataset

The Banknote Authentication dataset is used for Decision Tree and linear SVM experiments.

- **Samples:** 1,372
- **Input features:** 4
- **Target:** Binary class label (`0` or `1`)
- **Features:**
  - Variance of Wavelet Transformed Image
  - Skewness of Wavelet Transformed Image
  - Curtosis of Wavelet Transformed Image
  - Entropy of Image

The dataset is split into training and testing sets using an 80:20 ratio, with stratification to preserve class proportions.

### 2. Make Moons Dataset

The synthetic `make_moons` dataset is used to study nonlinear classification.

- **Samples:** 500
- **Input features:** 2
- **Target:** Binary class label
- **Noise:** 0.10
- **Random state:** 42

This dataset helps demonstrate why nonlinear kernels can outperform linear decision boundaries on complex data.

## Implementation

### Part A: Decision Trees

The Decision Tree classifier is implemented from scratch using NumPy.

Key components include:

- Gini impurity calculation
- Evaluation of candidate splits
- Selection of the best feature and threshold
- Recursive tree construction
- Prediction using the constructed tree
- Entropy and information gain calculations

The scratch implementation is compared with `DecisionTreeClassifier` from scikit-learn. Experiments also investigate tree depth, splitting criteria, and two-dimensional decision boundaries.

### Part B: Linear SVM from Scratch

A linear Support Vector Machine is implemented using NumPy.

The objective function combines L2 regularization and hinge loss:

\[
J(w,b)=\frac{1}{2}\|w\|_2^2+
\frac{C}{n}\sum_{i=1}^{n}
\max(0,1-y_i(w^Tx_i+b))
\]

The implementation includes:

- Feature standardization
- Conversion of labels from `0, 1` to `-1, +1`
- Hinge loss and objective calculation
- Subgradient calculation
- Parameter updates using gradient-based optimization
- Prediction using the learned decision boundary

Experiments investigate the effects of iteration count and regularization parameter \(C\). The scratch model is compared with scikit-learn's `LinearSVC`.

### Part C: Nonlinear SVM Kernels

The following SVM kernels are evaluated on the `make_moons` dataset:

- **Linear kernel:** Creates a linear decision boundary.
- **Polynomial kernel:** Models nonlinear relationships through polynomial feature interactions.
- **RBF kernel:** Models flexible nonlinear boundaries using distance-based similarity.

Decision boundary plots and experiments with different RBF gamma values demonstrate how kernel choice and hyperparameters influence model flexibility and classification performance.

### Part D: Final Model Comparison

Five models are compared on the Banknote Authentication test set:

1. Scratch Decision Tree
2. Scikit-learn Decision Tree
3. Scratch Linear SVM
4. Scikit-learn Linear SVM
5. Scikit-learn RBF SVM

The evaluation considers:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Training time
- Prediction time

## Results

The following results were obtained on the selected 80:20 test split of the Banknote Authentication dataset.

| Model | Accuracy | Weighted F1-score |
|---|---:|---:|
| Scratch Decision Tree | 98.55% | 98.55% |
| Scikit-learn Decision Tree | 99.27% | 99.27% |
| Scratch Linear SVM | 97.09% | 97.10% |
| Scikit-learn Linear SVM | 98.55% | 98.55% |
| Scikit-learn RBF SVM | 100.00% | 100.00% |

The RBF SVM achieved the highest test accuracy in this experiment. However, this result applies to the selected test split and does not guarantee perfect performance on unseen data.

The scikit-learn implementations trained faster than the scratch implementations in the recorded experiments, reflecting the efficiency of optimized library implementations compared with the Python-based educational implementations.

## Technologies Used

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create a virtual environment (optional)

```bash
python -m venv .venv
```

Activate it:

**macOS/Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 4. Add the dataset

Place `data_banknote_authentication.txt` in the dataset location expected by the notebook. If the notebook uses a relative path, ensure the working directory matches the path specified in the code.

### 5. Run the notebook

```bash
jupyter notebook
```

Open the lab notebook and execute the cells sequentially.

## Key Learnings

- Decision Trees classify samples using recursive, feature-based splits.
- Gini impurity and entropy are alternative criteria for measuring class impurity.
- SVMs learn decision boundaries by maximizing the margin while penalizing margin violations.
- Feature standardization is important for SVMs when features have different scales.
- The parameter \(C\) controls the trade-off between regularization and hinge-loss penalties.
- Nonlinear kernels allow SVMs to learn decision boundaries that linear models cannot represent.
- Scratch implementations help explain algorithm mechanics, while scikit-learn provides optimized implementations.
- Model selection should ideally use cross-validation, with a separate holdout test set reserved for final evaluation.

## Conclusion

This lab demonstrates the mathematical foundations and practical behavior of Decision Trees and Support Vector Machines through scratch implementations, visualizations, and comparisons with scikit-learn. The experiments highlight the importance of model complexity, feature scaling, regularization, and kernel selection. Although the RBF SVM achieved 100% accuracy on the selected test split, cross-validation and independent evaluation would be needed to establish how reliably it generalizes to unseen data.

## Author

DS605 — Fundamentals of Machine Learning  
Lab Assignment 8
