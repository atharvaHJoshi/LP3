# ML Final — Master Viva Question Bank

A single compiled list of all viva questions for the five practicals, topic
by topic. Short answers are provided for rapid revision; the detailed
explanations behind each answer live in [`README.md`](README.md).

- [General ML basics (cross-cutting)](#general-ml-basics-cross-cutting)
- [Practical 1 — Linear Regression (Uber)](#practical-1--linear-regression-uber)
- [Practical 2 — KNN & SVM (Email spam)](#practical-2--knn--svm-email-spam)
- [Practical 3 — ANN (Bank churn)](#practical-3--ann-bank-churn)
- [Practical 4 — Gradient Descent](#practical-4--gradient-descent)
- [Practical 5 — K-Means Clustering](#practical-5--k-means-clustering)
- [Comparison & wrap-up questions](#comparison--wrap-up-questions)

---

## General ML basics (cross-cutting)

1. **What is Machine Learning?** — Programs that improve from experience
   (data) without being explicitly reprogrammed (Arthur Samuel).
2. **AI vs ML vs DL?** — AI is the umbrella goal; ML learns patterns from
   data; DL is ML with multi-layer neural networks.
3. **Supervised vs unsupervised learning?** — Supervised: labeled data
   (predict y). Unsupervised: no labels (find structure).
4. **Classification vs regression?** — Discrete labels vs continuous values.
5. **Where do your 5 practicals fit?** — P1 regression, P2–P3 supervised
   classification, P4 optimization (training engine), P5 unsupervised
   clustering.
6. **Parametric vs non-parametric models?** — Fixed parameter count vs
   parameters/data that grow with the dataset; KNN is non-parametric.
7. **Lazy vs eager learning?** — KNN defers all work to prediction time
   (lazy); linear/SVM/ANN build a model at training (eager).
8. **What is bias? Variance?** — Systematic error from oversimplification vs
   sensitivity of the model to the training sample.
9. **Explain the bias–variance trade-off.** — Complexity ↑ lowers bias, raises
   variance; total error = bias² + variance + noise; aim for the sweet spot.
10. **What is overfitting / underfitting? How do you spot and fix them?** —
    Train≫test performance, or both high, respectively. More/less model
    capacity, regularization, early stopping, CV for detection.
11. **Why split data at all?** — Unbiased estimate of performance on unseen
    data; anything evaluated on training data is optimistic.
12. **Train/validation/test roles?** — Fit / choose model & hyperparameters /
    final one-time honest check.
13. **What is cross-validation? What does k-fold mean?** — Rotate holdout
    folds to get a stabler estimate; k = number of folds (typically 5/10).
14. **What is data leakage?** — Test-set information influencing training
    (scaler fitted on all data, peeking during feature selection). Avoided by
    pipeline order.
15. **Standardization vs normalization?** — z = (x−μ)/σ vs min-max scaling to
    [0,1]; standardize by default, min-max when bounded ranges needed.
16. **Why scale features?** — Distance-based models (KNN, SVM, K-Means) and
    gradient descent misbehave when features have wildly different ranges.
17. **Which practicals required scaling and why?** — P2 (KNN + SVM), P3 (ANN
    gradients), P5 (K-Means distances). P1 linear regression is scale-
    invariant in predictions, but scaling aids optimization.
18. **One-hot vs label encoding?** — Nominal categories need independent
    binary columns; integer codes invent a false order.
19. **What does random_state=42 do?** — Fixes the RNG so splits/initializations
    are reproducible run-to-run.
20. **What is a pipeline in sklearn and why use it?** — Chaining
    scaler→model so test data transforms only with training-fitted params.

---

## Practical 1 — Linear Regression (Uber)

21. **What did your notebook do step by step?** — Load `uber.xls` → inspect
    (head/tail) → dropna + drop_duplicates → fit `LinearRegression` on
    pickup/dropoff coordinates + passenger count → predict fare for a sample.
22. **Why is fare prediction regression?** — The target is continuous.
23. **Equation of linear regression?** — y = b₀ + b₁x₁ + … + bₚxₚ.
24. **What does fit() compute?** — Coefficients minimizing the sum of squared
    residuals (OLS / normal equation).
25. **Cost function and formula?** — MSE = (1/n)Σ(y−ŷ)².
26. **Why squared error?** — Differentiable everywhere, penalizes large
    errors, unique analytic minimum.
27. **What are residuals?** — Individual prediction errors y − ŷ.
28. **MAE vs MSE vs RMSE?** — Magnitude, squared (unit²), square-rooted (back
    to original units).
29. **What is R² and interpretation of R² = 0.75?** — 75% of target variance
    explained by the features.
30. **Adjusted R²?** — Penalizes adding useless features, unlike plain R².
31. **Assumptions of linear regression?** — Linearity, independence,
    homoscedasticity, normal residuals, no multicollinearity.
32. **What is multicollinearity and its symptom?** — Correlated predictors →
    unstable coefficients, huge standard errors.
33. **What happens with outliers in OLS?** — They pull the line strongly
    because errors are squared.
34. **How would you improve this model?** — Feature engineering (ride
    distance!), log transform, Ridge/Lasso, polynomial features, proper
    train/test metrics.
35. **How is this connected to Practical 4?** — GD can minimize the same MSE
    iteratively instead of the closed form.

---

## Practical 2 — KNN & SVM (Email spam)

36. **What is the dataset?** — Word-frequency matrix of emails with a
    spam/ham label ("Email No." dropped as an ID).
37. **KNN — how does it classify?** — Majority vote among the k nearest
    training points measured by a distance.
38. **Why is KNN lazy/non-parametric?** — No abstraction built at train time;
    no distribution assumed.
39. **Training vs prediction cost of KNN?** — O(1) training, O(n·d) per
    prediction.
40. **Distance metrics used by KNN?** — Euclidean (default), Manhattan,
    Minkowski (generalization, p), Hamming (categorical).
41. **Why is Minkowski special?** — p=1 Manhattan, p=2 Euclidean.
42. **What if k is very small / large?** — k=1: noise-following overfit;
    huge k: over-smoothed underfit.
43. **Why prefer odd k in binary classification?** — Avoids tied votes.
44. **Why scale features before KNN?** — Distances get hijacked by
    large-range features otherwise.
45. **What is the curse of dimensionality?** — In thousands of dims,
    distances concentrate, k-neighbor information weakens.
46. **What are hyperplanes, margins and support vectors in SVM?** — The
    decision boundary w·x+b=0; the gap 2/‖w‖; the closest critical points.
47. **What does SVM optimize?** — Max margin ⇔ min ½‖w‖² (+ C·Σξᵢ in soft
    margin / hinge loss).
48. **Hard vs soft margin?** — Zero tolerance vs slack-penalized tolerance
    for messy data.
49. **What does C control?** — High C → fit hard, narrow margin, overfit
    risk; low C → wide margin, regularized.
50. **What is the kernel trick?** — Implicit high-dimensional mapping via
    kernel functions K(xᵢ,xⱼ), no explicit coordinates.
51. **Which kernels exist?** — Linear, polynomial, RBF, sigmoid.
52. **Which kernel was used and why?** — Linear — email features are
    high-dimensional and near-linearly separable (typical for text).
53. **What is gamma in RBF?** — Radius of influence of one training point;
    big γ → narrow spikes → overfit.
54. **What is hinge loss?** — max(0, 1 − y·(w·x+b)); zero for points beyond
    margin, linear penalty otherwise.
55. **KNN vs SVM in one line each?** — KNN votes from stored neighbors (lazy,
    O(n) at predict); SVM learns a compact margin model (eager, fast predict).
56. **Why scaled pipelines here too?** — Margin geometry is Euclidean, same
    scale sensitivity.
57. **What does classification_report print?** — Precision, recall, F1 and
    support per class + averages.
58. **Precision vs recall in spam terms?** — Precision: flagged spam that is
    truly spam (blocked real mail cost); recall: share of actual spam caught.
59. **Why F1 score?** — Harmonic mean balancing precision and recall.
60. **Which model performed better in your comparison and why?** — Check the
    printed accuracies; SVM typically edges out in high-dimensional text.

---

## Practical 3 — ANN (Bank churn)

61. **What is churn prediction?** — Predicting if a customer will exit the
    bank (Exited = 1) — binary supervised classification.
62. **Why drop RowNumber/CustomerId/Surname?** — IDs/names have no signal;
    keeping them encourages memorization.
63. **Why one-hot encode Geography/Gender?** — Nominal categories; integer
    labels would imply a false order.
64. **What does drop_first=True prevent?** — The dummy-variable trap
    (perfect multicollinearity between dummies).
65. **How many inputs reach the network and why?** — 11 = 8 numeric + 2
    Geography dummies + 1 Gender dummy.
66. **Describe the architecture.** — Sequential ANN: Dense(16, ReLU) →
    Dense(8, ReLU) → Dense(1, sigmoid).
67. **What is a neuron's computation?** — z = Σwᵢxᵢ + b; a = f(z).
68. **Why ReLU in hidden layers?** — Non-saturating gradient, cheap, sparse;
    avoids vanishing gradients of sigmoid/tanh.
69. **Why sigmoid on the output?** — Produces P(churn) ∈ (0,1) for binary
    classification, paired with binary cross-entropy.
70. **What are forward and backward propagation?** — Forward: compute
    activations layer by layer; backward: chain-rule gradients of the loss
    w.r.t. every weight.
71. **What loss is used and why not MSE?** — Binary cross-entropy; MSE with
    sigmoid outputs gives tiny/non-convex gradients.
72. **Write binary cross-entropy.** — −(1/N)Σ[y·log ŷ + (1−y)·log(1−ŷ)].
73. **What is the Adam optimizer?** — Momentum (1st moment) + RMSprop (2nd
    moment) + bias correction; adaptive per-parameter learning rates.
74. **Epoch vs batch vs iteration?** — Full pass over data / samples per
    update / one weight update (10 epochs × ~250 steps = ~2500 updates here).
75. **How do you compute steps per epoch?** ⌈N/batch⌉ ≈ 8000/32 = 250.
76. **Why scale inputs for a neural net?** — Balanced gradients; unscaled
    features slow or destabilize gradient descent.
77. **Why fit the scaler on training data only?** — Test statistics leaking
    into training = data leakage.
78. **What does stratify=y do in train_test_split?** — Preserves the ~80/20
    churn ratio in both subsets.
79. **Why is the churn dataset imbalanced and why does it matter?** — ~20%
    churners: accuracy alone can hide a model that never predicts churn.
80. **What is the confusion matrix layout?** — Rows = actual, columns =
    predicted: TN, FP / FN, TP.
81. **Interpret TP, TN, FP, FN in churn terms.** — Caught leaver, retained
    customer, false alarm, missed leaver.
82. **Which is worse for a bank, FP or FN?** — FN (lost customer); the
    threshold can be lowered to trade precision for recall.
83. **What is precision / recall / F1?** — TP/(TP+FP); TP/(TP+FN);
    2PR/(P+R).
84. **What is ROC-AUC?** — Separation ability across all thresholds.
85. **Why threshold at 0.5?** — Conventional balance point of the sigmoid
    output; tunable for cost-sensitive recall.
86. **What is the vanishing gradient problem?** — Saturation of sigmoid/tanh
    chains shrinks gradients layer by layer.
87. **How do you fight overfitting in ANNs?** — Dropout, L2, early stopping,
    more data, smaller nets.
88. **Why not initialize all weights to zero?** — Symmetry: every neuron
    learns identically. Random He (ReLU)/Xavier (tanh) init is used.
89. **Sequential vs Functional API?** — Line stack vs arbitrary graphs;
    this task needs only Sequential.
90. **How would you improve the churn model?** — class_weight/SMOTE, dropout
    + early stopping, LR tuning, evaluate with recall/AUC, feature
    engineering.

---

## Practical 4 — Gradient Descent

91. **What did your notebook implement?** — From-scratch GD minimizing
    f(x) = (x+3)² with update x ← x − 0.1·2(x+3), starting at x = 2.
92. **What is a gradient?** — Vector of partial derivatives; steepest ascent
    direction, so we step minus it.
93. **The update rule?** — θ ← θ − η·∂J(θ).
94. **What is the learning rate and its failure modes?** — Step size; too
    small = slow, too large = overshoot/diverge.
95. **Why f′(x) = 2(x+3)?** — Chain rule on (x+3)².
96. **Where does GD end up, and in how many steps roughly?** — x ≈ −3; error
    shrinks ×0.8 each step, so ~30–40 steps suffice even with 100 iterations.
97. **Why is f(x) = (x+3)² GD-friendly?** — Convex parabola — no local minima
    traps; convergence is provable here.
98. **Batch vs Stochastic vs mini-batch GD?** — Full dataset vs one sample vs
    batches (32–256): stability, speed/noise, balance.
99. **Which variant does Keras use?** — Mini-batch (batch_size=32) with Adam.
100. **What is momentum?** — Velocity = β·v + ∇J smoothed across steps;
     rolls past small bumps, accelerates consistent directions.
101. **Local vs global minima and saddle points?** — Better-than-neighbors /
     best overall / flat point that's min in some directions, max in others.
102. **Convergence criteria?** — Tiny gradient, tiny parameter change, tiny
     loss change, or max iterations.
103. **Why not use the normal equation for ANN training?** — Non-convex
     non-linear loss; no analytic solution.
104. **How does this practical connect to the ANN practical?** — Same
     optimization; Keras computes gradients via backprop and updates via
     Adam — your notebook does the update by hand in 1-D.
105. **What shape indicates healthy GD training?** — Smooth monotonic loss
     curve flattening near the minimum.

---

## Practical 5 — K-Means Clustering

106. **Why is K-Means unsupervised?** — It never sees labels; it groups by
     similarity only.
107. **State the algorithm.** — Init K centroids → assign to nearest →
     recompute means → repeat until stable.
108. **What is inertia/WCSS?** — Σ over clusters of squared distances of
     members to their centroid; the minimized objective.
109. **What is the elbow method?** — Inertia vs K plot; pick the bend where
     extra clusters stop paying.
110. **For your data, which K was chosen and why?** — 3 — the WCSS curve
     flattens beyond it.
111. **Why is scaling critical here?** — Euclidean distances: SALES would
     dominate unscaled features.
112. **What is K-Means++?** — Distance-aware centroid initialization; better
     local optima (sklearn default.)
113. **Why n_init=10?** — Ten restarts; keep the best inertia to dodge bad
     local optima.
114. **Does K-Means converge to the global optimum?** — No — only a local
     one, on every run.
115. **Distance used by K-Means?** — Euclidean.
116. **Limitations of K-Means?** — Pre-set K, spherical cluster assumption,
     outlier sensitivity, scale sensitivity, mixed (categorical) data.
117. **How do you handle outliers?** — Remove/cap them, K-Medoids, DBSCAN.
118. **What is the silhouette score and range?** — (b−a)/max(a,b) per point,
     s ∈ [−1,1]; higher = well-formed clusters; used to select K.
119. **What did you learn from value_counts and groupby means?** — Segment
     sizes and the business profile of each cluster (bulk / premium /
     routine orders).
120. **What is the scatter plot really showing?** — 2 of the 4 scaled
     dimensions, colored by cluster assignment.
121. **K-Means vs hierarchical clustering?** — Flat converged K partitions
     vs nested dendrogram, no pre-set K, O(n²) cost.
122. **What is DBSCAN good for?** — Arbitrary shapes, automatic outlier and
     noise detection, no K needed.
123. **Real applications of clustering?** — Customer segmentation, image
     color compression, document grouping, anomaly detection.
124. **What is hard vs fuzzy clustering?** — Point belongs to exactly one
     cluster vs degrees of membership (GMM / fuzzy C-means).

---

## Comparison & wrap-up questions

125. **KNN vs linear SVM — when would you choose each?** — Small/medium data
     and low dims, interpretable local behavior → KNN; high-dim text, big
     data, fast inference → linear SVM.
126. **Linear regression vs KNN regression?** — Global linear model with few
     coefficients vs local average over neighbors.
127. **Logistic regression vs your ANN?** — Linear decision boundary vs
     non-linear learned features; both output probabilities.
128. **Why does the ANN use gradient descent while K-Means doesn't?** — K-Means
     is directly minimized by iterative re-assignment of its (convex per
     step) objective; ANN has no closed-form or simple assignment update.
129. **Which practicals are supervised / unsupervised?** — P1, P2, P3
     supervised; P5 unsupervised; P4 is the optimization used by the
     supervised ones.
130. **One metric you would report for each practical?** — RMSE (P1);
     accuracy + F1 (spam, P2); confusion matrix + recall (churn, P3);
     final x/y and loss curve (P4); inertia + silhouette (P5).
131. **How would you improve every model?** — P1: distance feature,
     Ridge/Lasso; P2: CV-tuned k, C, γ + class balancing; P3: class weights,
     dropout, early stopping, AUC; P4: LR schedule/momentum; P5: silhouette
     -based K, outlier cleaning, K-Means++ init.
132. **Explain bias–variance using your own practicals.** — k=1-KNN overfits
     (variance); linear regression on non-linear churn (bias); elbow K too
     small over-simplifies clusters; 2-layer ANN is a middle complexity →
     decent balance.
133. **What does reproducibility mean in your notebooks?** — Fixed
     random_state everywhere so teacher-rerun gives identical numbers.
134. **If the examiner says "your accuracy is low" — how do you respond?**
     Diagnose: more data, features, capacity, better hyperparameters,
     class weighting, evaluate on the right metric — never claim tuning
     until metrics prove the split is fair.
135. **What is the single most common preprocessing mistake?** — Scaling or
     encoding before the train/test split (leakage).
