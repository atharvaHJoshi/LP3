# ML Final Practicals — Viva Preparation Guide

Complete viva-prep **README** for the five Semester 7 Machine Learning final
practicals. Every topic that the viva examiner can ask is covered in detail:
the theory behind each algorithm, a step-by-step walkthrough of exactly what
the notebook does, and a question-and-answer bank for that practical.

> The full **numbered list of viva questions** (all topics, quick one-line
> answers) is in [`VIVA_QUESTIONS.md`](VIVA_QUESTIONS.md).

## The five practicals at a glance

| # | Practical (notebook) | Algorithm | Task type | Dataset | Libraries used | Evaluation |
|---|----------------------|-----------|-----------|---------|----------------|------------|
| 1 | [`ass1-uber.ipynb`](ass1-uber.ipynb) | Linear Regression | Supervised — Regression | Uber ride data (pickup/dropoff coordinates, passenger count → fare) | `pandas`, `numpy`, `scikit-learn` | Predicted fare, regression concepts |
| 2 | [`ass02-knn-svm.ipynb`](ass02-knn-svm.ipynb) | K-Nearest Neighbors + Support Vector Machine | Supervised — Classification | Email dataset (word-frequency features) — spam / ham | `pandas`, `numpy`, `scikit-learn` | Accuracy, classification report (precision / recall / F1) |
| 3 | [`ass03-bank-customer-churn-neural-network.ipynb`](ass03-bank-customer-churn-neural-network.ipynb) | Artificial Neural Network (ANN) — 16 → 8 → 1 | Supervised — Binary classification | Bank customers (`Churn_Modelling.xls`) → will the customer exit? | `pandas`, `numpy`, `scikit-learn`, `tensorflow/keras` | Accuracy, confusion matrix |
| 4 | [`ass04-gradientdescent.ipynb`](ass04-gradientdescent.ipynb) | Gradient Descent (implemented from scratch) | Optimization (core of all training) | Objective function f(x) = (x + 3)² | `numpy`, `matplotlib` | Converged minimum (x, y) + plot |
| 5 | [`ass05-clustering.ipynb`](ass05-clustering.ipynb) | K-Means clustering (Elbow method) | **Unsupervised** — Clustering | Sales data (`sales_data_sample.xls`) | `pandas`, `numpy`, `matplotlib`, `scikit-learn` | WCSS / inertia, elbow curve, cluster profiles |

---

## Practical 1 — Uber Fare Prediction using Linear Regression
*(notebook: `ass1-uber.ipynb`)*

### 1.1 What the notebook actually does (code walkthrough)

1. **Load the data** — `pd.read_csv("uber.xls")`, then inspect it with
   `df.head()` (first 5 rows) and `df.tail(10)` (last 10 rows) to sanity-check
   that the columns are parsed correctly.
2. **Clean the data** — `df.dropna()` removes rows with missing values and
   `df.drop_duplicates()` removes repeated records. This is essential: a
   regression trained on duplicated rows is *biased toward those routes*, and
   NaN values would crash numeric computation.
3. **Model** — `LinearRegression()` from `scikit-learn` is trained on the
   features (pickup longitude, pickup latitude, dropoff longitude, dropoff
   latitude, passenger count) to predict the target (**fare amount**).
4. **Prediction** — a new sample,
   `arr = [[-73.98, 40.75, -73.97, 40.76, 2]]`
   (pickup lon, pickup lat, dropoff lon, dropoff lat, 2 passengers), is fed to
   `lr.predict(arr)` and the model prints the expected fare.

> The five numbers in the test sample are exactly the feature columns the
> `LinearRegression` model was fitted on, **in the same order**. Ordering and
> count of features in `predict()` must match training.

### 1.2 Theory — every topic in detail

#### What is Regression? What is Linear Regression?
- **Regression** is a supervised learning task where the output is a
  **continuous numeric value** (fare ₹, price $, temperature °C) instead of a
  class label.
- **Linear Regression** assumes the target is a **linear (straight-line /
  hyper-plane) combination** of the input features:

  - Simple linear regression (1 feature):  **y = m·x + b**
  - Multiple linear regression (p features): **ŷ = b₀ + b₁x₁ + b₂x₂ + … + bₚxₚ**

  Here `bᵢ` are the **learned coefficients** (slope, weight) and `b₀` is the
  **intercept** (bias).

#### How does it learn the best line? (Least Squares / OLS)
- Training = finding the coefficients that make the fitted line as close as
  possible to the data points.
- The **error (residual)** for one point is `(yᵢ − ŷᵢ)`.
- **Ordinary Least Squares (OLS)** chooses coefficients that minimize the
  **Sum of Squared Residuals**: `Σ(yᵢ − ŷᵢ)²`.
- Equivalent view — minimizing the **Mean Squared Error (MSE) cost function**:

  ```
  J(w, b) = (1/n) Σ (yᵢ − ŷᵢ)²
  ```

  Squaring (a) penalizes big mistakes heavily and (b) removes the sign so
  errors don't cancel out.
- Two ways to solve it:
  1. **Closed-form / Normal Equation:** `w = (XᵀX)⁻¹Xᵀy` (used by
     `sklearn`'s `LinearRegression`).
  2. **Iteratively with Gradient Descent** — exactly what Practical 4
     implements from scratch.

#### Why "least *squares*" and not absolute error?
MSE is differentiable everywhere and has a unique analytic minimum; both
properties make optimization easy. MAE is not differentiable at 0 and gives
less weight to outliers.

#### Evaluation metrics for regression (know all four)
| Metric | Formula | Notes |
|--------|---------|-------|
| **MSE** | mean of squared errors | Punishes large errors; units are squared |
| **RMSE** | √MSE | Same unit as target — most reported |
| **MAE** | mean of absolute errors | Robust to outliers |
| **R² (coefficient of determination)** | `1 − SS_res / SS_tot` | 1.0 = perfect fit, 0 = as good as predicting the mean, negative = worse than the mean |

#### Assumptions of linear regression (classic viva question)
1. **Linearity** — the true relationship between X and y is linear.
2. **Independence** — observations (residuals) are independent of each other.
3. **Homoscedasticity** — residual variance is constant across all X.
4. **Normality of residuals** — errors are normally distributed (needed for
   confidence intervals / hypothesis tests).
5. **No (strong) multicollinearity** — features should not be highly
   correlated with each other, otherwise coefficient estimates become unstable.

#### Important related concepts
- **Train-test split**: evaluate on data the model never saw → honest
  estimate of generalization.
- **Overfitting** in regression = fitting the noise (e.g., with high-degree
  polynomial features); fixed by more data, fewer features, or
  **regularization** — **Ridge (L2)** shrinks coefficients toward zero,
  **Lasso (L1)** can shrink some to exactly zero (feature selection).
- **Outliers** pull the least-squares line strongly toward themselves
  (squared error is sensitive) → hence the drop of bad data rows first.
- **Why the Uber problem suits linear regression**: fare ≈ base charge +
  per-km rate × ride distance; distance derived from coordinates → mostly
  linear in the features (plus small surcharges).

### 1.3 Viva questions & answers — Practical 1

1. **Why is Uber fare prediction a regression problem and not classification?**
   The target (fare) is a continuous number; classification would need discrete
   labels.
2. **What do before-fit cleaning steps (`dropna`, `drop_duplicates`) prevent?**
   Training crashes on NaN; duplicates make the model give extra (biased)
   weight to repeated routes.
3. **What does `LinearRegression().fit(X, y)` compute?**
   The least-squares optimum: coefficients that minimize Σ(y − ŷ)².
4. **Interpret a slope coefficient, e.g., dropoff-latitude = +45.**
   Holding all other features constant, each 1-unit increase in dropoff
   latitude raises the predicted fare by 45.
5. **What is the intercept?**
   The model's base prediction when every feature is 0 (a "base fare" idea).
6. **What is the cost function of linear regression?**
   MSE — mean of squared differences between actual and predicted values.
7. **How would you evaluate the trained model rigorously?**
   Keep a test set, compute MAE/RMSE/R² on it (not on training data).
8. **What is R² and what does R² = 0.75 mean?**
   75% of the variance in fare is explained by the selected features.
9. **Name the assumptions of linear regression.** Linearity, independence,
   homoscedasticity, normal residuals, no multicollinearity.
10. **What happens if two features are almost identical?**
    Multicollinearity — coefficients become unstable/ill-defined even though
    predictions may not change.
11. **How does gradient descent relate to linear regression?**
    Same MSE objective can be minimized iteratively with gradient descent
    (Practical 4) instead of the closed-form normal equation.
12. **What is overfitting here and two fixes?**
    Too many polynomial features fitting noise → simplify model / Ridge-Lasso
    regularization / more data.
13. **Why is RMSE preferred for reporting over MSE?**
    Its unit matches the fare units, so it is directly interpretable.
14. **What if the true relation is non-linear?**
    Apply PolynomialFeatures, use a different model (tree/GBM) or transform
    variables — plain linear regression will underfit.

---

## Practical 2 — Email Spam Classification using KNN and SVM
*(notebook: `ass02-knn-svm.ipynb`)*

### 2.1 What the notebook actually does (code walkthrough)

1. **Load** the email dataset (`emails.xls`) — each row is one email; the
   columns are mostly **word-frequency counts** (how many times each keyword
   appeared) plus a final **`Prediction`** column = the label (`spam`/`ham`).
   An ID column **`Email No.`** is dropped because it carries no information.
2. **Clean** — `dropna()` + `drop_duplicates()` (an email appearing twice is
   not two samples).
3. **Split features / label** — `X = df.drop('Prediction', 1)` (word counts),
   `y = df['Prediction']`.
4. **Train-test split** — 80% train / 20% test, `random_state=42` makes the
   split reproducible (same every run).
5. **KNN model** — built inside a `Pipeline`:
   ```
   StandardScaler() → KNeighborsClassifier(n_neighbors=5)
   ```
   The **scaler is fitted on training data only** and applied to the test
   data at predict time — that prevents **data leakage**.
6. **SVM model** — `make_pipeline(StandardScaler(), SVC(kernel='linear'))` —
   a **linear-kernel** Support Vector Classifier with default `C = 1`.
7. **Evaluation** — `accuracy_score` and a full `classification_report`
   (precision, recall, F1 per class), then both models are compared.

### 2.2 Theory — K-Nearest Neighbors (KNN)

#### Idea
Classify a new email by looking at the **k emails most similar to it** and
taking a **majority vote** of their labels. "Similar" = closest in feature
space. For regression tasks, KNN would instead average the neighbors' values.

#### Key properties (know these cold)
- **Lazy / instance-based learner**: KNN does **not** build a model at
  training time; it just stores all the training points ("memorizes"). All
  computation happens at predict time.
- **Non-parametric**: no assumption about the data distribution; the decision
  boundary can be arbitrarily complex (it is determined by the data normals).
- Training cost ≈ O(1). Prediction cost = O(n·d) where n = number of training
  samples, d = number of features — KNN is **slow at prediction, fast at
  training** (the exact opposite of most parametric models).

#### The distance metrics (which one measures "nearest"?)
| Metric | Formula | Use case |
|--------|---------|----------|
| **Euclidean** (default) | `√Σ(aᵢ−bᵢ)²` | Generic continuous features |
| **Manhattan** | `Σ|aᵢ−bᵢ|` | High-dimensional data, robust to outliers |
| **Minkowski** | `(Σ|aᵢ−bᵢ|ᵖ)^(1/p)` | Generalization: p=2 → Euclidean, p=1 → Manhattan |
| **Hamming** | # positions where symbols differ | Categorical / text data |

The generalization formula of Euclidean/Manhattan distance is the **Minkowski
distance** with parameter p.

#### Choosing k — the art of the algorithm
- **k too small (k=1)**: decision follows single noise points → jagged
  boundaries, **high variance → overfitting**.
- **k too large**: smooths over local structure, neighboring classes blend →
  **high bias → underfitting**.
- Common practice: **odd k** (5, 7, …) for binary problems so the vote cannot
  tie; choose k by cross-validation. This notebook fixes **k = 5**.
- **Weighted KNN** can vote with distance weights (`weights='distance'` in
  sklearn) — closer neighbors count more.

#### Why `StandardScaler` before KNN? (asked almost every time)
KNN is a **distance-based** algorithm. Words such as "FREE" may have counts in
the hundreds while rare words have counts ≈ 0 — the large-range features then
**dominate the Euclidean distance**, and the algorithm effectively ignores
everything else. Standardization `z = (x − mean) / std` puts every feature on
mean 0 / std 1 so all words contribute fairly, and distances become meaningful.

#### The curse of dimensionality (classic follow-up)
With thousands of word-count dimensions, points become **sparse** and pair-wise
distances **converge** (everything is nearly equally far from everything).
Recent-neighbor information then gets less discriminative → accuracy drops.
Remedies: feature selection, PCA / dimensionality reduction.

### 2.3 Theory — Support Vector Machine (SVM)

#### The core idea
SVM is a **maximal-margin classifier**. Among all hyperplanes that separate
"spam" and "ham", it picks the one whose **distance to the nearest points of
each class is the largest**, because a wide margin generalizes better than a
narrow one.

- The separating hyperplane: **w·x + b = 0**
- The margin (total, both sides): **2 / ‖w‖** → maximizing the margin is
  equivalent to **minimizing ‖w‖²/2** subject to correct classification.
- **Support vectors** = the few points lying exactly on the margin edges.
  They alone "support" (define) the boundary — delete any non-support vector
  and the model is unchanged; move/delete a support vector and the boundary
  moves.

#### Hard margin vs soft margin (the `C` parameter)
- **Hard margin**: no training point may violate the margin — only works when
  the data is perfectly separable; brittle with noise.
- **Soft margin** allows violations with **slack variables ξᵢ** and a penalty
  **C** added to the objective:
  `min ½‖w‖² + C·Σξᵢ`
  - **Small C** → tolerates more mistakes, **wider margin, stronger
    regularization, less overfitting**.
  - **Large C** → tries hard to classify every training point correctly,
    **narrow margin, risks overfitting**. (Default in sklearn: `C = 1`.)

#### Kernels and the kernel trick (favorite question)
Some data is **not linearly separable** (e.g., a circle inside a ring).
The **kernel trick** implicitly projects data into a higher-dimensional space
where a linear boundary exists — **without ever computing the high-dimensional
coordinates** (only inner products `K(xᵢ,xⱼ)` are needed).

| Kernel | sklearn value | When |
|--------|---------------|------|
| Linear | `kernel='linear'` | High-dimensional, many features (text/spam) |
| Polynomial | `kernel='poly'`, degree d | Feature-interaction boundaries |
| **RBF / Gaussian** (default) | `kernel='rbf'` | Generic non-linear boundaries |
| Sigmoid | `kernel='sigmoid'` | Neural-flavor kernels |

Two knobs for RBF: **gamma** = how far each single training point "reaches"
(large γ → narrow spikes → overfit; small γ → smooth, underfit), and **C**
(margin penalty). Why `kernel='linear'` fits this spam data well: word-count
features are already extremely high-dimensional (one dimension per word),
spaces where classes are often linearly separable and linear SVMs are
fast + strong (this is exactly why linear kernels are the standard for text).

#### Hinge loss
A moderate-margin classifier optimizes the **hinge loss**
`L = max(0, 1 − y·(w·x + b))` with an L2 penalty on ‖w‖ for the soft margin —
equals 0 for correctly-classified points beyond the margin, and grows linearly
for violations.

#### Why SVM also needs feature scaling
The margin is measured in **Euclidean distances**, so features with big
numeric ranges would control the geometry — hence the same
`StandardScaler` pipeline.

#### Quick head-to-head (great closing answer)
| | KNN | Linear SVM |
|---|-----|-----------|
| Learning type | Lazy, instance-based | Eager — builds a parametric model |
| Storage | Entire training set kept | Only w, b + support vectors |
| Train / predict cost | Train ≈ 0; Predict O(n) | Train heavy; Predict O(#SV·d) |
| Sensitive to | k, distance metric, scale | C, kernels, scaling |
| High-dimensional text | Suffers (curse of dimensionality) | Works very well |
| Output | Vote of k neighbors | Decision function (margin), no native probability |

### 2.4 Evaluation — precision / recall / F1 (must know in spam context)

- **Accuracy** = (TP + TN) / total — how often the classifier is right overall.
  Fine when classes are balanced; misleading when they are not (a "never
  spam" model can be 95% accurate on a 5% spam dataset).
- **Precision (spam)** = TP / (TP + FP) — *of the mails we flagged spam, how
  many truly were?* Low precision → **real mail wrongfully blocked** (costly!).
- **Recall (spam)** = TP / (TP + FN) — *of all actual spam, how much did we
  catch?* Low recall → spam sneaks into the inbox.
- **F1** = harmonic mean `2·(P·R)/(P+R)` — balances both.
- The **classification_report** prints these per class plus macro/weighted
  averages and the **support** (number of true samples of each class).

### 2.5 Viva questions & answers — Practical 2

1. **Is KNN parametric or non-parametric? Lazy or eager?**
   Non-parametric and lazy — it stores data, no model is fitted.
2. **Why must KNN use standardized features?**
   Euclidean distance is scale-sensitive; unscaled big-range features dominate.
3. **What happens when k is too small / too large?**
   Small k → overfits (noise votes); large k → underfits, boundary too smooth.
4. **How does KNN predict?**
   Compute distance to all training points, take the k closest, majority vote
   (classification) or mean (regression).
5. **Which distance metrics does KNN use?**
   Euclidean (default), Manhattan, Minkowski generalization, Hamming for
   categorical.
6. **Why is Minkowski mentioned with p?**
   p=2 gives Euclidean, p=1 gives Manhattan — one formula, two metrics.
7. **What is the training cost of KNN? Prediction cost?**
   Near-zero training; O(n·d) per prediction — expensive at scale.
8. **What is a hyperplane in SVM?**
   The decision boundary w·x + b = 0 separating the classes.
9. **What are support vectors?**
   The training points lying closest to the boundary that alone define the
   maximal-margin hyperplane.
10. **What is the margin and how is it computed?**
    Distance between the two margin hyperplanes = 2/‖w‖; maximized by SVM.
11. **Hard vs soft margin?**
    Hard allows zero violations (perfectly separable data); soft margin adds
    slacks ξ with penalty C and tolerates noise.
12. **What does C control?**
    Trade-off between a wide margin and classifying all points: small C =
    regularized/wide, large C = aggressive/narrow (overfit-prone).
13. **What is the kernel trick?**
    Using kernel functions of inner products to reach an implicit
    high-dimensional linear separation without explicit coordinate mapping.
14. **Which kernels exist? Which was used here and why?**
    Linear, polynomial, RBF, sigmoid; here `linear` because email word-count
    features are high-dimensional and approximately linearly separable.
15. **What are γ and C in the RBF kernel?**
    γ = radius of influence of each training point; C = misclassification
    penalty. Both high → overfit; both low → underfit.
16. **What loss does SVM optimize?**
    Hinge loss with L2 regularization of the weights.
17. **Which model won in your run and why might that be?**
    Compare printed accuracies; typical outcome: SVM similar-or-better than
    KNN because it does not suffer the dimensionality of ≤ 3000 word features
    and ignores irrelevant directions via the margin.
18. **Why pipelines?**
    Fit the scaler on train only, reuse for test → clean code and no leakage.
19. **What does the classification_report give you?**
    Precision, recall, F1, and support per class plus averaged scores.
20. **What is the "curse of dimensionality"?**
    In thousands of dimensions distances concentrate; neighbors become less
    meaningful, hurting distance-based models like KNN.

---

## Practical 3 — Customer Churn Prediction using an ANN (Keras)
*(notebook: `ass03-bank-customer-churn-neural-network.ipynb`)*

### 3.1 What the notebook actually does (code walkthrough)

1. **Load** `Churn_Modelling.xls` — 10,000 bank customers, target column
   `Exited` (1 = customer left the bank, 0 = stayed).
2. **Drop useless columns** `RowNumber`, `CustomerId`, `Surname` — unique IDs
   and names carry **zero predictive signal**; keeping them would make the
   model memorize rows.
3. **Encode categoricals** —
   `pd.get_dummies(df, columns=['Geography','Gender'], drop_first=True, dtype=int)`
   creates binary indicator columns (`Geography_Germany`, `Geography_Spain`,
   `Gender_Male`); `drop_first=True` drops one category per feature to avoid
   the **dummy-variable trap** (multicollinearity). 11 numeric features remain.
4. **Split** — 80/20 with **`stratify=y`**: the exit/no-exit class ratio is
   preserved in both halves — crucial because churn data is **imbalanced**
   (~20% churners).
5. **Scale** — `StandardScaler().fit_transform(X_train)`,
   `sc.transform(X_test)` — scaler fitted on train only (no leakage).
6. **Build the network** (Sequential API):
   ```
   Dense(16, activation='relu', input_dim=11)   ← hidden layer 1
   Dense(8,  activation='relu')                 ← hidden layer 2
   Dense(1,  activation='sigmoid')              ← output
   ```
   A classic **funnel/bottleneck** shape: 11 inputs → 16 → 8 → 1 probability.
7. **Compile** — `optimizer='adam'`, `loss='binary_crossentropy'`,
   `metrics=['accuracy']`.
8. **Train** — `model.fit(..., epochs=10, batch_size=32)`: the whole training
   set is passed 10 times, in mini-batches of 32 rows
   (≈ 8000/32 = **250 weight-update steps per epoch** — mini-batch GD).
9. **Predict & threshold** — `y_pred = (model.predict(X_test) > 0.5)`: the
   sigmoid outputs a probability; it becomes class 1 when p > 0.5.
10. **Evaluate** — `accuracy_score` + `confusion_matrix`.

### 3.2 Theory — Artificial Neural Networks, every topic

#### What is a neuron / what is an ANN?
Inspired (loosely) by biological neurons. An artificial neuron computes a
**weighted sum plus bias, then a non-linear activation**:

```
z = Σ wᵢ·xᵢ + b        a = f(z)
```

Layers of such neurons: **input layer** (the 11 features) → **hidden layer(s)**
(16 and 8 ReLU neurons — where non-linear feature combinations are learned) →
**output layer** (1 sigmoid neuron — churn probability).

Why non-linear activations? Without them, stacked linear layers collapse into
one big linear function — the network could never learn complex boundaries.

#### Activation functions — the table to reproduce in the viva
| Function | Formula | Range | Where | Note |
|----------|---------|-------|-------|------|
| **Sigmoid** | 1/(1+e⁻ᶻ) | (0,1) | binary output layer | Vanishing gradient for large ‖z‖ |
| **Tanh** | (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ) | (−1,1) | hidden (older nets) | Zero-centered |
| **ReLU** | max(0, z) | [0,∞) | hidden layers (default) | Cheap, sparse, fights vanishing gradient |
| **Leaky ReLU** | z<sub>(z>0)</sub> else αz | (−∞,∞) | hidden | Fixes "dying ReLU" |
| **Softmax** | e^{zⱼ}/Σe^{zₖ} | (0,1), sums to 1 | multi-class output | Generalizes sigmoid |

#### Why ReLU in the hidden layers and sigmoid at the output?
- **ReLU hidden** — non-saturating for positive z (gradient = 1), fast to
  compute, sparse activations → faster, more stable training than sigmoid's
  gradients (≤ 0.25) that multiply down the layers ("vanishing gradient").
- **Sigmoid output** — squashes z into (0,1), read as
  **P(Exited = 1)**; pairs with **binary cross-entropy** loss. (The threshold
  0.5 then turns the probability into the class decision.)

#### Forward propagation
Input → weigh/sum → activation, layer by layer, until the output probability.
Pure computation, no learning yet.

#### Backpropagation (one-line rule + a bit more)
Backprop computes **∂Loss/∂(every weight)** efficiently by applying the
**chain rule backwards** from the loss through the layers, using cached
activations. Those gradients are what **gradient descent** consumes to update
the weights: `w ← w − η·∂L/∂w`. Forward-prop caches activations; backprop
produces gradients; the optimizer updates — that loop *is* training.

#### Loss functions for classification
- **Binary cross-entropy** (this notebook):
  `L = −(1/N) Σ [ y·log ŷ + (1−y)·log(1−ŷ) ]`
  Punishes confident wrong predictions enormously (log → −∞ as ŷ→wrong extreme).
- **Categorical cross-entropy** — one-hot multi-class with a softmax output.
- **Why not MSE for classification?** With a sigmoid output the MSE
  landscape is non-convex and its gradient becomes tiny when the neuron is
  saturated → slow, gradient-vanishing training.

#### Optimizers (Adam = adaptive momentum)
| Optimizer | Core idea |
|-----------|-----------|
| **SGD** | Plain updates using gradients of mini-batches; often noisy but generalizes well |
| **SGD + Momentum** | Adds an exponentially-weighted velocity term, damps oscillations, accelerates |
| **RMSprop** | Divides the LR by a running average of squared gradients → per-parameter adaptive step |
| **Adam** (used here) | **Momentum (1st moment) + RMSprop (2nd moment)** with bias correction; default lr = 0.001; generally the safe first choice |

#### Epochs, batch size, steps — the relationship
- **Epoch** = one complete pass over the full training set (here 10 epochs).
- **Batch size** = rows per weight update (here 32 — *mini-batch* GD).
- **Steps/epoch** = ⌈N / batch⌉ ≈ 250 for 8,000 train rows.
- **Iterations/updates** = epochs × steps/epoch.
- Bigger batch: smoother gradients, more memory; smaller batch: noisier but
  regularizing. 32 is a robust default.

#### Why scale features before an ANN? (top viva question)
Unscaled features (e.g., EstimatedSalary ~ 0…200,000 with CreditScore ~ 0…1000)
make the loss surface elongated — gradient descent zig-zags and converges
slowly; large-magnitude inputs also saturate sigmoid-type units early.
StandardScaler → mean 0, std 1 → well-conditioned gradients. **Fit the scaler
on the training data only** — scaling using test statistics is **data
leakage**.

#### One-hot vs label encoding & the dummy trap
- **Label encoding** would assign France=0, Spain=1, Germany=2 → invents a
  **false ordinal order**; neural nets read the numbers and learn bogus
  relations. **One-hot** gives each category an independent binary column.
- `drop_first=True` removes one level per feature (France here): one column
  would be a **perfect linear combination** of the rest → **multicollinearity
  ("dummy variable trap")**. With 3 levels you need 2 dummies.

#### Imbalanced classes — why accuracy is not enough here
Only ~20% of customers churn. A model that always predicts "0" gets ~80%
accuracy while catching **zero churners**. Hence `stratify=y` on the split and
the **confusion matrix**:

```
              Predicted 0    Predicted 1
Actual 0   |    TN        |      FP      |
Actual 1   |    FN        |      TP      |
```

- **Recall (churn) = TP/(TP+FN)** — how many real churners we caught (a bank
  wants recall high; missing a leaver is a lost revenue stream).
- **Precision (churn) = TP/(TP+FP)** — how many alerts were really churners
  (wasted retention offers target precision).
- **F1** balances the two; **ROC-AUC** measures class separation across all
  thresholds.

###### Why `stratify=y` matters
Without it a random split could (by chance) put fewer churners in the test
set, making metrics unstable and incomparable across runs.

#### Overfitting / underfitting in neural nets — and the fixes
- Symptoms: train accuracy rises while validation stalls/falls.
- Fixes: **Dropout**, **L2 regularization**, **early stopping**, more data,
  simpler architecture; report validation metrics, not training metrics.

#### Initialization & gradient pathologies (senior-level extras)
- **Vanishing gradients** — sigmoid/tanh saturation multiplies gradients < 1
  layer by layer → deep nets stop learning (a reason ReLU is chosen).
- **Exploding gradients** — opposite failure; mitigated by gradient clipping,
  better init.
- **Initialization** — small random values; **He init** pairs with ReLU,
  **Xavier/Glorot** with tanh/sigmoid. All-zero init fails (all neurons learn
  identically — symmetric problem).

### 3.3 Viva questions & answers — Practical 3

1. **What is churn prediction?**
   Predicting whether a customer will leave (Exited = 1) — supervised binary
   classification.
2. **Why remove RowNumber / CustomerId / Surname?**
   Unique per row, no predictive signal; keeping them invites memorization.
3. **Why one-hot encode Geography/Gender instead of label encoding?**
   They are nominal; integers would impose a false order; one-hot keeps
   categories independent.
4. **What does drop_first=True do and why?**
   Drops one dummy level per feature to avoid the dummy-variable trap
   (perfect multicollinearity).
5. **How many input features reach the network and why 11?**
   8 numeric + 2 geography dummies + 1 gender dummy = 11.
6. **Describe the architecture.**
   11-16-8-1: two ReLU hidden layers and a single sigmoid output neuron.
7. **Why sigmoid in the output layer?**
   It outputs P(churn) in (0,1) and pairs with binary cross-entropy loss.
8. **Why ReLU in hidden layers?**
   Non-saturating gradients, cheap, sparse — avoids vanishing gradients.
9. **What is backpropagation?**
   Chain-rule-based computation of the loss gradient w.r.t. every weight.
10. **What loss is used and why that and not MSE?**
    Binary cross-entropy — logarithmic loss suited to probabilities; MSE with
    a sigmoid output saturates and converges badly.
11. **What optimizer is used and how does Adam work?**
    Adam — momentum + adaptive per-parameter learning rates (first and second
    moments with bias correction).
12. **What are epochs and batch_size here — how many updates per epoch?**
    10 full passes; batches of 32; ≈ 250 steps/epoch (8,000 rows).
13. **Why must features be scaled? Why fit the scaler on train only?**
    Gradient conditioning + avoiding data leakage from test statistics.
14. **What does stratify=y do?**
    Preserves the churn/no-churn class ratio in train and test splits.
15. **What does `predict > 0.5` mean?**
    The sigmoid probability is thresholded at 0.5 to produce the label.
16. **Interpret the top-left / bottom-right cells of the confusion matrix.**
    TN (correct non-churn) and TP (correct churn catches).
17. **Which error is worse for a bank — FP or FN?**
    Both costly: FN misses a real leaver (no retention), FP wastes an offer on
    a happy customer — trade-off controlled by the threshold.
18. **This dataset is ~80/20 imbalanced — is accuracy sufficient?**
    No; report precision/recall/F1 or AUC as well.
19. **What would you do to improve the model?**
    Class weights / resampling (SMOTE), dropout + early stopping, feature
    engineering, tuning epochs/lr, AUC-based selection.
20. **How is a Sequential model different from the Functional API?**
    Sequential = linear stack (single-input/single-output); Functional
    supports shared/multiple branches — not required for this simple net.

---

## Practical 4 — Gradient Descent from Scratch (f(x) = (x+3)²)
*(notebook: `ass04-gradientdescent.ipynb`)*

### 4.1 What the notebook actually does (code walkthrough)

```python
def gradient_descent(derivative, x, lr, iterations):
    for i in range(iterations):
        x = x - lr * derivative(x)      # the ONLY update rule
    return x

def f(x):          return (x+3)**2        # objective (the "loss")
def derivative(x): return 2*(x+3)         # its derivative f'(x)

minimum = gradient_descent(derivative, x=2, lr=0.1, iterations=100)
```

- Function to minimize: **f(x) = (x+3)²** — a parabola whose exact minimum is
  at **x = −3** (f(−3) = 0). This choice is deliberate: the true answer is
  known, so **convergence can be verified by eye**.
- Start at **x = 2**, learning rate 0.1, up to 100 iterations.
- Each iteration: **x ← x − 0.1 · 2(x+3)** — move a step **opposite the
  gradient** (downhill).
- The result is printed (`Minimum x ≈ −3`, `Minimum y ≈ 0`) and the parabola is
  plotted, with the found minimum marked **red**.

#### Trace of a few steps (worth reproducing on paper in the viva)
| step | x | f′(x) = 2(x+3) | update: x ← x − 0.1·f′ |
|------|------|------|----------|
| start | 2.000 | 10.0 | 2 − 1.0 = 1.0 |
| 1 | 1.000 | 8.0 | 0.2 |
| 2 | 0.200 | 6.4 | −0.44 |
| … | … | … | each step multiplies the *distance to −3* by **0.8** (= 1 − 2·lr) |

Because x−(−3) shrinks geometrically ×0.8, after ~30–40 steps the point is
indistinguishable from −3, e.g. after 100 iterations the leftover error is
5·0.8¹⁰⁰ ≈ 10⁻⁹. The loop continues harmlessly (gradient ≈ 0 ⇒ steps ≈ 0) —
a stopped-clock situation: **the check "am I at the minimum?" is exactly
'gradient = 0'**.

### 4.2 Theory — gradient descent in full detail

#### What gradient descent is and why ML needs it
Gradient descent is an **iterative first-order optimization** algorithm that
moves a point oppositely to the local gradient until it reaches a
(local) minimum of a function. In ML, "the function" is the **loss** of your
model (MSE of linear regression, cross-entropy of a neural network), "the
point" is the **parameter vector** (weights), and learning = running GD on it.
This notebook isolates the idea in 1-D: minimize a parabola by walking it.

#### The update rule (one formula to write on the board)

```
θ_new = θ_old − η · ∂J/∂θ          (or in vector form: θ ← θ − η∇J(θ))
```

- **η (eta, "learning rate")** = step size.
- Why **minus**: ∂J/∂θ points toward *steepest ascent*; we want descent.
- Why "*f′(x)*" here: derivative of the 1-D loss; in a neural net the
  "derivative" is the gradient vector obtained by **backpropagation**
  (Practical 3).

#### The learning rate — the single most important knob
- **Too small** (η = 0.001): tiny steps → slow, may need millions of
  iterations, can stall before reaching the minimum.
- **Too large** (η = 1.2): the step **overshoots** to the other side, and with
  the ×|1 − 2·η| ratio > 1 each step the iterates **diverge** — the loss
  oscillates and explodes (NaNs in practice).
- **Just right**: geometric shrink of the error per step (here exactly
  ×0.8 per iteration with η = 0.1 on this loss).
- Practical schemes: try 0.1/0.01/0.001, plot the loss curve; decay η when
  progress plateaus (step decay, 1/t, cosine); **adaptive optimizers** adjust
  η automatically (RMSprop, Adam — again Practical 3!).

#### When does it stop? (convergence criteria)
Gradient norm < ε (stationary point), change in θ/loss < ε, or max iterations.
Here the cap is `iterations = 100`; the first two conditions are implicit
because the steps become numerically negligible.

#### Variants of gradient descent (name all three)
| Variant | Data used per update | Pros | Cons |
|---------|----------------------|------|------|
| **Batch GD** | entire dataset | exact/stable path | costly per step; stuck in local minima |
| **Stochastic (SGD)** | one random sample | cheap, noisy — can escape saddle points/local minima | zig-zaggy path |
| **Mini-batch GD** | batch of 32–256 | best of both — Keras default | needs batch-size tuning |

Aim: faster updates, smoother than pure SGD. Modern practice: mini-batch +
momentum/Adam.

#### Local minima, global minima, saddle points
- f(x) = (x+3)² is **convex** → every local minimum is the global one, and GD
  provably converges (with suitable η). This is why linear regression's MSE is
  nice.
- Deep networks are **non-convex**: many local minima, saddle points, plateaus.
  In practice large over-parameterized nets find good minima, and methods like
  momentum, SGD noise, LR schedules, random restarts help escape bad regions.
- **Local minimum**: better than all neighbors; **global**: best overall;
  **saddle**: gradient = 0 but neither — a maximum in one direction, minimum
  in another (the enemy of pure GD in high dimensions).

#### Momentum and the physics picture
Momentum accumulates the past gradients in a velocity term
`v = βv + ∇J; θ ← θ − ηv` — behaves like a heavy ball: it rolls over small
local bumps and accelerates along consistent directions. (Adam ≈ momentum +
per-direction adaptive scaling.)

#### Gradient descent family vs closed-form (OLS)
For *linear regression* the MSE is convex & quadratic — solvable exactly via
the Normal Equation **provided** you can invert XᵀX. Gradient descent scales to
any differentiable loss, huge data, and non-linear models (that is the only
option in deep learning — likely a "why not normal equation in ANN?" question:
activations make the loss non-convex/non-quadratic and there is no analytic
inverse).

### 4.3 Viva questions & answers — Practical 4

1. **What is gradient descent?**
   Iterative optimization moving parameters against the gradient to minimize
   a loss function.
2. **What is a gradient?**
   The vector of partial derivatives of the loss — the direction of steepest
   ascent (so we move minus it).
3. **Write the update rule.**
   θ ← θ − η·∂J/∂θ.
4. **What does the learning rate control and what if it's too big/small?**
   Step size; too big → overshoot/diverge; too small → slow or stall.
5. **Why f′(x) = 2(x+3)?**
   Chain rule on (x+3)²: d/dx[(u)²] = 2u·u′ with u = x+3.
6. **What is the true minimum and what did GD find?**
   x = −3, y = 0 — the printout converges to ≈ −3.
7. **How quickly does the error shrink here?**
   Distance to −3 multiplies by 0.8 each step (1 − 2·lr = 0.8).
8. **Difference between batch, stochastic and mini-batch GD?**
   Full-dataset vs one-sample vs small-batch gradients per update —
   stability vs speed trade-offs.
9. **Which variant does the ANN practical use?**
   Mini-batch: batch_size = 32 with the Adam optimizer.
10. **What is a local minimum? A saddle point?**
    Best-in-neighborhood point; and a flat point that is min in some
    directions, max in others.
11. **Why is this particular loss easy for GD?**
    It's convex — a single parabola, so no wrong valleys exist.
12. **Momentum — what and why?**
    A running average (velocity) of past gradients that smooths and
    accelerates the path.
13. **When would you prefer the Normal Equation over GD?**
    Small-to-medium, linear problems with invertible XᵀX; never for deep nets.
14. **How is GD in this notebook connected to training the ANN in Practical 3?**
    Same math: keras computes gradients by backprop and Adam performs the
    θ ← θ − η·step updates — this notebook does it by hand in 1-D.
15. **What is a loss curve and what shape indicates good training?**
    Loss vs iterations; a smooth monotone decay flattening out.

---

## Practical 5 — K-Means Clustering (Elbow Method) on Sales Data
*(notebook: `ass05-clustering.ipynb`)*

### 5.1 What the notebook actually does (code walkthrough)

1. **Load** `sales_data_sample.xls` and inspect with `head()`, `tail(10)`,
   `info()`, `columns`, `shape`, `isnull().sum()` — a full mini-EDA.
2. **Feature selection** — 4 numeric business features:
   `QUANTITYORDERED, PRICEEACH, SALES, MSRP` (`dropna().copy()` removes
   incomplete rows).
3. **Scale** — `StandardScaler().fit_transform(data)`: all four features now
   mean 0 / std 1.
4. **Elbow method** — for K = 1…10:
   ```
   KMeans(n_clusters=i, random_state=42, n_init=10).fit(X)
   wcss.append(model.inertia_)
   ```
   then a `plt.plot` of **K vs WCSS (inertia)** shows the "elbow" — the K
   beyond which adding clusters yields little inertia gain. **The elbow here
   ≈ K = 3.**
5. **Final clustering** — `KMeans(n_clusters=3, random_state=42, n_init=10)`
   is fit and each row assigned with `fit_predict`; the labels are stored in
   `data['Cluster']`.
6. **Profile clusters** — `value_counts()` (cluster sizes) and
   `groupby('Cluster').mean()` (per-cluster feature means → interpreting what
   makes each segment, e.g., "big bulk orders" vs "low-volume orders").
7. **Visualize & save** — scatter of QUANTITYORDERED vs SALES colored by
   cluster (`cmap='rainbow'`), result exported to `clustered_sales.csv`.
   (Note: the scatter is a **2-D slice** of the 4-D clustering space.)

> Why n_init=10: k-means may converge to a **different local optimum** for
> different initial centroids. `n_init=10` runs the fit 10 times with
> different seeds and keeps the run with the lowest inertia. `random_state=42`
> makes the initialization reproducible.

### 5.2 Theory — unsupervised learning and K-Means

#### Supervised vs unsupervised (must say this clearly in the viva)
- Practical 1–3 were **supervised**: labeled target (fare; spam; exited…).
- This one is **unsupervised**: **no labels at all** — the algorithm's goal is
  to **discover structure**: partitions of similar records ("clusters") such
  that within-cluster similarity is high and between-cluster similarity low.

#### Clustering and where K-Means sits
Families of clustering algorithms:
1. **Partitioning** — K-Means, K-Medoids (PAM) — need K in advance.
2. **Hierarchical** — agglomerative/divisive, dendrogram, cut to choose K.
3. **Density-based** — DBSCAN — arbitrary shapes + outlier detection.
4. **Model-based** — GMM with EM — soft/probabilistic clusters.
K-Means is the classic partitioning member: fast, scalable, and the workhorse
of segmentation tasks (here: customer/order segments; also: market basket,
image compression, document grouping).

#### The K-Means algorithm — 4 steps to recall verbatim
1. **Initialize** K centroids (randomly chosen points; **K-Means++** spreads
   them away from each other instead — the sklearn default `init='k-means++'`
   improves both quality and speed of convergence).
2. **Assignment step**: assign every point to its **nearest centroid**
   (Euclidean distance) — this carves the space into Voronoi regions.
3. **Update step**: recompute each centroid as the **mean** of the points
   assigned to it.
4. **Repeat 2–3** until no reassignments occur / centroids stop moving /
   inertia falls below tolerance; it always converges (finite WCSS decrease
   each cycle) but only to a **local optimum** — hence N inits.

Why "K-**Means**"? The step-3 update is literally the mean of each cluster.

#### The objective — inertia / WCSS (know both names and the formula)
K-Means minimizes the **Within-Cluster Sum of Squares**:

```
WCSS (inertia) = Σᵢ Σ_{x ∈ Cᵢ} ‖x − μᵢ‖²
```

i.e., the total squared distance of points to their assigned centroids.
On scaled 4-D data, `model.inertia_` returned this value for every K tried.

#### Choosing K: the elbow method (and its alternative)
- As K grows inertia **mechanically decreases** (K = n gives WCSS = 0).
- The **elbow** is the K where the curve visibly bends: below it, each added
  cluster kills a lot of inertia; above it, gains are marginal → more clusters
  just split good groups.
- Read here: elbow at **K = 3** → sales split into three natural
  order-volume/value segments.
- Alternative: **Silhouette score**
  `s = (b − a) / max(a, b)` per point where a = mean intra-cluster distance,
  b = mean distance to best neighboring cluster; s ∈ [−1,1], pick the K with
  the highest average s (measures cohesion vs separation).

#### Numerical pitfalls (nearly guaranteed viva follow-ups)
- **Feature scaling is mandatory**: revenue `SALES` spans tens of thousands
  while `PRICEEACH` spans ~tens; unscaled, the Euclidean distance is
  dominated by `SALES` alone and clusters form around that single feature.
  Standardization equalizes all four features' influence.
- **Outliers** drag the mean (centroid) — K-Means is not robust: one huge
  order can become its own cluster. Options: cap/remove outliers, use
  **K-Medoids** (medoid = actual member point, robust).
- **K must be pre-chosen**; wrong K gives forced groupings (silhouette helps).
- **Local optima**: bounded via `n_init` reruns and K-Means++ initialization.
- **Spherical/convex assumptions**: equal-variance, round blobs — fails on
  elongated or moon-like shapes (DBSCAN is the fix), and struggles with
  mixed categorical data (that's K-Prototypes territory).

#### Reading the output (how you'd interpret the result in the viva)
- `value_counts()`: relative sizes (a segment with 5 orders is suspicious —
  possibly outlier-driven).
- `groupby('Cluster').mean()`: e.g., cluster A = high quantity × high unit
  price → bulk B2B orders; cluster B = low quantity × high price → premium
  retail; cluster C = low value→ small routine orders. This last, human step
  is what turns clustering into a **business segmentation**.

### 5.3 Viva questions & answers — Practical 5

1. **Why is this practical "unsupervised"?**
   No target column is used; clusters are found purely from feature
   similarity.
2. **State the K-Means algorithm steps.**
   Initialize K centroids → assign points to nearest → recompute centroids as
   means → repeat until stable.
3. **What does the K stand for?**
   The number of clusters specified before training.
4. **Which features were used and why those?**
   QUANTITYORDERED, PRICEEACH, SALES, MSRP — direct numerical order/business
   value measurements.
5. **Why is scaling required before K-Means?**
   Euclidean distance; an unscaled big-range feature (SALES) would dominate.
6. **What is WCSS / inertia?**
   Sum of squared distances of points to their cluster centroid; K-Means
   minimizes it.
7. **Explain the elbow method.**
   Plot inertia vs K; pick the bend where marginal inertia reduction becomes
   small — here K = 3.
8. **Why did the elbow say 3 for your data?**
   Beyond 3 clusters the WCSS curve flattens, showing no meaningful new
   segments.
9. **What is K-Means++?**
   An initialization spreading initial centroids apart (probabilistically
   proportional to squared distance) — better and more stable convergence.
   (sklearn default.)
10. **Why n_init=10 and random_state=42?**
    10 independent initializations to avoid bad local optima; a fixed seed
    for reproducibility of the reported result.
11. **Does K-Means always converge? To the global optimum?**
    It converges in finitely many steps, but only to a local optimum.
12. **What distance does K-Means use?**
    Euclidean (by default).
13. **Limitations of K-Means?**
    Pre-set K, sensitive to outliers and scale, assumes spherical clusters,
    local optima, struggles with non-numeric features.
14. **How would you handle outliers in clustering?**
    Remove/cap them, use K-Medoids, or a density method like DBSCAN.
15. **How do you interpret the scatter plot of QUANTITYORDERED vs SALES?**
    It's a 2-D projection of a 4-D clustering — you see three rainbow-colored
    regions, not full evidence.
16. **What does cluster mean-values profiling tell you?**
    The business character of each segment (bulk vs premium vs routine
    orders) — the actual deliverable.
17. **What is the silhouette score?**
    (b−a)/max(a,b) ∈ [−1, 1]; cohesion vs separation measure to pick K
    instead of (or with) the elbow.
18. **Alternatives to the elbow to choose K?**
    Silhouette, Davies–Bouldin score, gap statistic, business constraints.
19. **K-Means vs hierarchical clustering?**
    Flat K clusters vs nested dendrogram trees; the latter doesn't predetermine
    K but is O(n²)-heavy for big data.
20. **Practical applications of K-Means?**
    Customer segmentation, image compression/colors, document grouping,
    anomaly pre-screening, initial semi-supervised labeling.

---

## Cross-cutting viva topics (usually the first 5 minutes)

These general ML questions open almost every viva — be fluent in them.

### Definitions ladder
- **Artificial Intelligence** — machines performing tasks requiring human-like
  intelligence; **Machine Learning** — subset of AI: systems that *learn from
  data* instead of being explicitly programmed (Arthur Samuel's definition);
  **Deep Learning** — subset of ML: multi-layer neural networks learning
  hierarchical features automatically (Practical 3).
- **Supervised learning** — labeled data, learn mapping X → y:
  *regression* (continuous y — Practical 1) and *classification* (discrete y —
  Practical 2, 3).
- **Unsupervised learning** — unlabeled data, find structure: clustering
  (Practical 5), dimensionality reduction, association rules. GD
  (Practical 4) is the optimization engine *behind* most supervised learning.
- **Parametric vs non-parametric** — fixed number of parameters
  (linear regression, logistic regression, ANN-size given) vs growing with the
  data (KNN, decision trees, SVM with kernels).
- **Instance-based** — KNN memorizes training examples; SVM stores only
  support vectors; ANN summarizes everything into weights.

### Bias–variance trade-off
- **Bias** = systematic error from over-simplifying (high bias ⇒ **underfitting**;
  training AND test error both high).
- **Variance** = sensitivity to the particular training sample (high variance
  ⇒ **overfitting**; training error low, test error high).
- Total error = bias² + variance + irreducible noise; increasing model
  complexity lowers bias and raises variance; choose the sweet spot.
- Per practical: k=1 KNN (high variance), linear regression on non-linear
  data (high bias), 10-epoch 16-8-1 ANN may slightly underfit complex churn
  patterns, elbow-K = 3 (variance of clustering with tiny K too high).

### Splits & validation
- **Train / test** split (80/20 here): fit vs honest generalization estimate.
- **Validation set / CV (k-fold, e.g., 5 or 10)**: model/hyperparameter
  selection *without touching the test set*; k-fold rotates the validation
  portion for a stabler estimate (KNN's k and SVM's C/γ are chosen by CV).
- **Stratified split** keeps class proportions (Practical 3's `stratify=y`).
- **Data leakage** — any test-set information (scaler stats, dummies, feature
  selection) entering training; pipelines prevent it.

### Preprocessing checklist (order matters!)
1. Drop ID/noise columns → 2. handle missing values (drop / impute mean /
   median / mode) → 3. remove duplicates → 4. encode categoricals (one-hot
   for nominal, ordinal encoding for ordered) → 5. split FIRST → 6. scale
   (fit on train only) → 7. (optionally balance classes).
- **Standardization** z=(x−μ)/σ vs **Min-Max** (x−min)/(max−min) → [0,1]:
  standardization is unbounded/robust-default; min-max keeps shapes for
  neural nets/images.
- **Why scale**: distance models (KNN, K-Means, SVM) and gradient descent
  (ANN) all depend on feature magnitudes.

### Metrics master list
- Regression: MAE, MSE, RMSE, R², adjusted R².
- Classification: accuracy, **precision**, **recall**, F1, confusion matrix,
  specificity, ROC curve & AUC; per-class balance matters when imbalanced.

### Library cheat-sheet (what to attribute to which library)
- **NumPy** — arrays, vector math (Practical 4's GD loop).
- **Pandas** — DataFrames: load (`read_csv`), inspect (`head/tail/info/shape/
  isnull`), clean (`dropna/drop_duplicates/drop`), encode (`get_dummies`),
  aggregate (`groupby`, `value_counts`).
- **Matplotlib** — plots (GD parabola, elbow curve, cluster scatter).
- **scikit-learn** — `train_test_split`, `StandardScaler`, `Pipeline/`
  `make_pipeline`, models (`LinearRegression`, `KNeighborsClassifier`, `SVC`,
  `KMeans`), metrics (`accuracy_score`, `classification_report`, 
  `confusion_matrix`).
- **TensorFlow / Keras** — `Sequential`, `Dense`, `compile`, `fit`, `predict`.
- **`random_state=42`** — reproducibility: identical random splits and
  initializations on every run; **n_init** — K-Means restarts (Practical 5).

---

## Viva exam-day flow (what typically happens)

1. **Explain your code, practically line by line**: name the dataset, the
   features, the target, the algorithm, and the exact metric/plot printed.
2. **Rapid theory rounds per topic**: definitions → one formula each
   (MSE, margin 2/‖w‖, BCE loss, GD update rule, WCSS) → properties of
   hyperparameters (k, C, γ, η, K, batch, epochs) → scaling/encoding reasons.
3. **Compare pairs**: KNN vs SVM, one-hot vs label encoding, standardization
   vs normalization, batch vs SGD vs mini-batch, K-Means vs hierarchical,
   R² vs RMSE, precision vs recall.
4. **"Why this and not X?"**: why SVM for spam; why stratify; why
   drop_first; why sigmoid not softmax (binary!); why not the normal equation
   for ANN; why scaling before K-Means.
5. **Wrap-up questions**: improve each model, limitations of each algorithm,
   define bias–variance with a sketch of the U-curve.

Full question list with short answers: **[`VIVA_QUESTIONS.md`](VIVA_QUESTIONS.md)**
