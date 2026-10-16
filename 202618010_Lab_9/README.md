# DS605 Lab 7: Image Representation, PCA, Eigenfaces and Face Recognition

Dataset: AT&T (ORL) Database of Faces, 40 subjects x 10 images, 112x92 grayscale.

PCA/eigenface computations are implemented with NumPy linear algebra only (no `sklearn.decomposition.PCA`). Scikit-learn is used only in Part D for the classifier comparison.

## Workflow
Images -> grayscale pixel matrix -> flatten -> mean centering -> SVD / eigendecomposition -> explained variance -> eigenfaces -> reduced representation -> reconstruction -> nearest-neighbor matching -> ML classification

## How to run
1. Place the dataset in `data/` with subfolders `s1` ... `s40`, each holding 10 images.
2. `pip install -r requirements.txt`
3. Run `lab7_eigenfaces.ipynb` top to bottom.

## Repository contents
- `lab7_eigenfaces.ipynb`: full code with outputs
- `figures/`: `mean_face.png`, `explained_variance.png`, `eigenfaces.png`, `reconstructions.png`, `recognition_matches.png`
- `ml_comparison.csv`: final classifier comparison table

## Part A: Image representation
- 400 images, each 112x92 with intensities 0-255, flattened to 10,304-dim vectors, giving a 400 x 10,304 matrix with subject labels.
- Reproducible stratified split (NumPy RNG, seed 42): 7 train / 3 test images per subject, giving 280 train and 120 test images with all 40 subjects in both sets and no shared images.
- This is closed-set identification, so every subject must appear in training. The mean face and PCA are fitted on the training set only, and the test set is centered with the training mean.

## Part B: PCA, eigenfaces, compression
- Mean face = average of the 280 training images (one global mean).
- **SVD vs eigendecomposition:** the full 10,304x10,304 covariance matrix would need about 850 MB, so the eigendecomposition used the 280x280 Gram-matrix trick (eigenvectors mapped back with Xc^T v and normalized). Both routes agree: max eigenvalue difference over the top 279 components was 1.9e-9, and the minimum absolute cosine between matching eigenvectors was 0.999999999999996 (signs may differ, so absolute values were compared). Eigenvalues relate to singular values by lambda = s^2/(n-1).
- Only 279 components are non-zero (last singular value ~1e-11), because centering removes one degree of freedom.
- Explained variance: PC1 17.8%, PC2 12.6%, PC3 7.1%, PC4 5.7%, PC5 5.1%.

| k | 10 | 25 | 50 | 92 | 100 | 150 |
|---|---|---|---|---|---|---|
| Cumulative variance | 60.3% | 73.7% | 82.9% | 90.04% | 90.9% | 95.1% |

- **Minimum k for >= 90% variance: 92.** This is used in Parts B and C.
- **Reconstruction (test set MSE):** k=10: 658.7, k=25: 488.7, k=50: 396.2, k=92: 335.5, k=150: 299.4. Error falls quickly at first and then flattens (-49% from k=10 to 92, only -11% from 92 to 150). It does not reach 0 because test faces are not in the span of the training eigenfaces. RMSE at k=92 is about 18 gray levels (of 255).
- **Storage:** original 280 x 10,304 = 2,885,120 floats. PCA (scores + eigenfaces + mean face) = 280x92 + 92x10,304 + 10,304 = 984,032 floats, about 2.9x smaller overall. Per image, 10,304 numbers become 92 (about 112x smaller). The overall saving is modest on a small dataset because the eigenface matrix itself must be stored; the per-image saving grows in value as the dataset grows.

## Part C: Recognition in PCA space
Each test face is projected with the training mean and eigenfaces, and the nearest training face by Euclidean distance gives the predicted identity.

| Representation | Features | Accuracy |
|---|---|---|
| Raw pixels | 10,304 | 97.5% (117/120) |
| PCA (k=92) | 92 | 97.5% (117/120) |

PCA matched raw-pixel accuracy with about 112x fewer numbers per image. Euclidean distance in the top components approximates full pixel-space distance, so PCA here buys compactness and speed rather than accuracy.

Accuracy vs k (nearest neighbor, test set; observation only, not used to choose k): k=10: 95.0%, 25: 96.7%, 50: 98.3%, 92: 97.5%, 100: 97.5%, 150: 98.3%. Differences beyond k~50 are one image (0.83%) and are noise.

**Error analysis.** The 3 errors were test subjects 5 (matched to 40), 10 (matched to 23) and 32 (matched to 2): faces with a wide smile or a tilted/turned head. Correct matches had a median distance of about 1540 (90th percentile 2655). The error distances were 1985, 3156 and 3322.
- Nearest true-subject photo vs nearest wrong match: subject 5: 2415 vs 1985 (gap ~430, the test image looked different from all 7 of that person's training photos); subject 10: 3226 vs 3156 (gap 69); subject 32: 3350 vs 3322 (gap 27). For subjects 10 and 32 the correct identity lost by a very small margin.
- With only 7 training photos per person, some test expressions and poses are not covered, which is a limitation of pixel-based similarity.

## Part D: Traditional machine learning
Models: KNN, Logistic Regression, Linear SVM on PCA scores. All hyperparameters (k, C, scaling, n_neighbors, weights) were chosen with 5-fold stratified cross-validation on the training set only. The test set was used once, for the final report.

**Initial CV (train only, features standardized for LogReg/SVM):**

| k | KNN (k=1) | LogReg | Linear SVM |
|---|---|---|---|
| 10 | 0.8929 | 0.8679 | 0.9214 |
| 25 | 0.9429 | 0.9393 | 0.9357 |
| 50 | 0.9429 | 0.9464 | 0.9536 |
| 92 | 0.9393 | 0.8893 | 0.8571 |
| 100 | 0.9393 | 0.8714 | 0.8429 |
| 150 | 0.9393 | 0.7679 | 0.6893 |

Standardizing the PCA scores made LogReg/SVM accuracy fall as k grew, because it gives low-variance noise components the same weight as the leading ones. Unscaled KNN was stable. Tuning treated scaling as a setting, and both LogReg and SVM selected no scaling.

**Tuned CV:** KNN 0.9464 (k=35, n_neighbors=1, uniform), LogReg 0.9643 (k=50, C=0.01, no scaling), Linear SVM 0.9643 (k=35, C=0.001, no scaling).

**Final comparison (test set, 120 images):**

| Model | Components k | Test accuracy | Correct |
|---|---|---|---|
| Nearest neighbor (Part C) | 92 | 97.5% | 117/120 |
| KNN | 35 | 96.7% | 116/120 |
| Logistic Regression | 50 | 100% | 120/120 |
| Linear SVM | 35 | 97.5% | 117/120 |

Logistic Regression was best on this split, but the margin is 3-4 images, and it was tied with the SVM in cross-validation, so the models are close. The best models use 35-50 components, fewer than the 92 needed for 90% variance, so a compact representation is enough for good recognition.

## Limitations
- One train/test split (seed 42) on 120 test images; one image is 0.83%, so small differences are not significant.
- PCA was fitted on all 280 training images, not refit inside each CV fold. No test data was used, but a stricter protocol would refit per fold.
- The best C values for LogReg and SVM were at the low edge of their grids, so the search may not be fully converged.
- ORL contains several similar shots per person, which makes accuracy high; results may not carry over to harder datasets.