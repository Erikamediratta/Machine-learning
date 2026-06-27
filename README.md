
## Table of Contents
- [Part 1 — ML Fundamentals](#part-1--ml-fundamentals)
- [Part 2 — Regression Algorithms](#part-2--regression-algorithms)
- [Part 3 — Classification Algorithms](#part-3--classification-algorithms)
- [Part 4 — Ensemble Learning](#part-4--ensemble-learning)
- [Part 5 — Feature Engineering](#part-5--feature-engineering)
- [Part 6 — Data Preprocessing](#part-6--data-preprocessing)
- [Part 7 — Model Evaluation](#part-7--model-evaluation)
- [Part 8 — Explainable AI](#part-8--explainable-ai)
- [Part 9 — Hyperparameter Optimization](#part-9--hyperparameter-optimization)
- [Part 10 — Deep Learning Basics](#part-10--deep-learning-basics) *(includes Vanishing Gradient Problem)*
- [Part 11 — Complete ML Pipeline](#part-11--complete-ml-pipeline)

---

# Part 1 — ML Fundamentals

### What is Machine Learning?
Algorithms that learn patterns from data to make predictions, instead of following hardcoded rules. Performance improves as more data is seen.

### Types of ML
- **Supervised**: labeled data → predict output (classification/regression)
- **Unsupervised**: no labels → find structure (clustering, PCA)
- **Reinforcement**: agent learns via rewards/penalties from an environment

### Supervised vs Unsupervised vs Reinforcement
| | Data | Goal | Example |
|---|---|---|---|
| Supervised | Labeled | Predict output | Spam detection |
| Unsupervised | Unlabeled | Find structure | Customer segmentation |
| Reinforcement | Reward signal | Maximize reward | Game-playing AI |

### Train / Validation / Test Split
- **Train**: fit model parameters
- **Validation**: tune hyperparameters, pick best model
- **Test**: final unbiased performance check (touched only once)

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# Result: 70% train, 15% val, 15% test
```

### Cross Validation (K-Fold)
Splits data into K folds, trains K times (each fold held out once as validation), averages the results. More reliable than one train/val split, especially on small data.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(scores.mean())
```

### Cost Function
Measures how wrong predictions are. Training = minimizing this.
- Regression: MSE = `(1/n) * Σ(actual - predicted)²`
- Classification: Log Loss / Cross-Entropy

### Gradient Descent
Iterative optimization that updates parameters in the direction that reduces the cost function.

```python
# from-scratch gradient descent for linear regression
w, b = 0.0, 0.0
lr = 0.01
for epoch in range(1000):
    y_pred = w * X + b
    dw = -2 * np.mean(X * (y - y_pred))
    db = -2 * np.mean(y - y_pred)
    w -= lr * dw
    b -= lr * db
```

**Variants:** Batch GD (all data per step), Stochastic GD (1 sample per step, noisy but fast), Mini-batch GD (small batches — used in practice/deep learning).

### Learning Rate
Step size in gradient descent. Too high → overshoots/diverges. Too low → slow convergence, may get stuck.

### Bias vs Variance
- **Bias**: model too simple, misses real pattern → underfitting
- **Variance**: model too sensitive to training data/noise → overfitting
- Tradeoff: reducing one usually increases the other.

### Underfitting vs Overfitting
| | Train Error | Test Error | Fix |
|---|---|---|---|
| Underfitting | High | High | More features, less regularization, bigger model |
| Overfitting | Low | High | More data, regularization, simpler model, early stopping |

```python
# Quick diagnostic
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
# train high + test low -> overfitting
# both low -> underfitting
```

### Data Leakage
Information from outside the training set (future data, test set, or a feature only available *after* the target is known) leaks into training, inflating validation scores but failing in production.

**Common causes:** scaling/imputing before train-test split, using a feature that's a proxy for the target, not respecting time order in time-series.

```python
# WRONG — leakage: scaler sees test data
scaler.fit(X)  
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled, y)

# RIGHT — fit scaler on train only
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```
# Part 2 — Regression Algorithms

### Linear Regression
Fits a straight line minimizing squared error between predicted and actual values.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)
print(model.coef_, model.intercept_)
```
✅ Simple, interpretable | ❌ Assumes linearity, sensitive to outliers

### Multiple Linear Regression
Same idea, multiple input features: `y = b0 + b1*x1 + b2*x2 + ...`
Watch for **multicollinearity** (correlated features inflate/destabilize coefficients) — check with VIF.

### Polynomial Regression
Adds powers of features (x², x³) to capture curved relationships, while staying linear in the coefficients.

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
```
❌ High degree → overfits easily

### Ridge Regression (L2)
Adds penalty `λ * Σ(coef²)` to shrink coefficients toward zero (never exactly zero). Good with multicollinearity.

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0).fit(X_train, y_train)
```

### Lasso Regression (L1)
Adds penalty `λ * Σ|coef|` — can shrink coefficients to **exactly zero** → automatic feature selection.

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1).fit(X_train, y_train)
print(model.coef_)  # some will be exactly 0
```

### ElasticNet
Combines L1 + L2: `λ1*Σ|coef| + λ2*Σ(coef²)`. Use when many correlated features, some irrelevant.

```python
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train)
```

### MAE vs MSE vs RMSE vs R²
| Metric | Formula | Notes |
|---|---|---|
| MAE | mean(\|actual − pred\|) | robust to outliers |
| MSE | mean((actual − pred)²) | penalizes big errors heavily |
| RMSE | sqrt(MSE) | same units as target |
| R² | 1 − (SS_res / SS_tot) | % variance explained, 0–1 |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)
```

### Regularization (L1 vs L2) — Quick Recall
- **L1 (Lasso)**: sparse solution, drops features
- **L2 (Ridge)**: shrinks all coefficients, keeps all features
- Both reduce overfitting by penalizing large weights; strength controlled by `alpha`/`λ`.
# Part 3 — Classification Algorithms

### Logistic Regression
Predicts probability via sigmoid: `P = 1 / (1 + e^-z)` where `z = w.x + b`. Thresholded (commonly 0.5) for class label. Trained via log loss, not MSE.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)
```

### K-Nearest Neighbors (KNN)
Classifies a point by majority vote of its K closest neighbors (by distance). No training phase ("lazy learner").

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
```
❌ Needs feature scaling. Slow at predict time on large datasets. Hurt by curse of dimensionality.

### Decision Tree
Splits data on feature thresholds to maximize purity (Gini/Entropy) at each step, until a stopping rule.

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
```
- **Gini Impurity**: `1 - Σ(p_i²)`
- **Entropy**: `-Σ(p_i * log2(p_i))`
- ❌ Overfits if grown too deep; unstable (small data change → different tree)

### Random Forest
Bagging ensemble of many decision trees (bootstrapped data + random feature subsets per split), combined by majority vote/average.

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, max_depth=10).fit(X_train, y_train)
print(model.feature_importances_)
```

### Naive Bayes
Probabilistic classifier using Bayes' theorem, assuming features are conditionally independent given the class. Fast, works well for text.

```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train_counts, y_train)
```
`P(class|features) ∝ P(class) * Π P(feature_i | class)`

### Support Vector Machine (SVM)
Finds the hyperplane that maximizes the margin between classes.

```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0).fit(X_train, y_train)
```

### Kernel Trick
Implicitly maps data to a higher dimension (without explicitly computing it) so a linear separator works on non-linearly-separable data. Common kernels: `linear`, `poly`, `rbf`.

### Hard Margin vs Soft Margin
- **Hard margin**: zero tolerance for misclassification — only works if data is perfectly separable (rare in practice)
- **Soft margin**: allows some violations, controlled by `C` — high `C` = less tolerance (risk overfit), low `C` = more tolerance (risk underfit)
# Part 4 — Ensemble Learning

### Ensemble Methods
Combine multiple models to get better, more stable predictions than any single model. Two main families: **Bagging** (parallel, reduces variance) and **Boosting** (sequential, reduces bias).

### Bagging
Train models independently on bootstrapped (random, with-replacement) samples; average/vote results. Reduces variance.

```python
from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(n_estimators=50).fit(X_train, y_train)
```

### Boosting
Train models sequentially; each new model corrects errors of the previous ones. Reduces bias, but more prone to overfitting noisy data.

### Random Forest
(See Part 3) — Bagging applied to decision trees with random feature subsets per split.

### AdaBoost
Boosting where misclassified samples get **higher weights** for the next weak learner (usually a decision stump).

```python
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=100).fit(X_train, y_train)
```

### Gradient Boosting
Boosting where each new model fits the **residual errors** of the combined ensemble so far (gradient descent in function space).

```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3).fit(X_train, y_train)
```

### XGBoost
Optimized, regularized Gradient Boosting: faster, handles missing values natively, built-in L1/L2 regularization.

```python
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    reg_lambda=1.0, reg_alpha=0.0
)
model.fit(X_train, y_train)
```

### Random Forest vs XGBoost
| | Random Forest | XGBoost |
|---|---|---|
| Training | Parallel trees | Sequential trees |
| Reduces | Variance | Bias (+ variance via regularization) |
| Tuning needed | Minimal | Significant |
| Typical accuracy | Good | Often higher |
| Missing values | Needs handling | Native support |

### AdaBoost vs Gradient Boosting
| | AdaBoost | Gradient Boosting |
|---|---|---|
| Correction mechanism | Re-weight misclassified samples | Fit residual errors directly |
| Loss function | Exponential loss (classification) | Any differentiable loss |
| Sensitivity to outliers | High | Lower (with regularization) |
# Part 5 — Feature Engineering

### Feature Engineering
Creating/transforming input variables to make patterns more visible to the model (e.g., extracting `hour_of_day` from a timestamp).

```python
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['is_weekend'] = pd.to_datetime(df['timestamp']).dt.dayofweek >= 5
```

### Feature Selection
Choosing the most relevant subset of features, dropping noise/redundant ones — reduces overfitting, speeds training, improves interpretability.

### Filter Methods
Select features using stats vs. the target (correlation, chi-square, mutual info) — fast, model-agnostic, ignores feature interactions.

```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=10).fit(X_train, y_train)
X_selected = selector.transform(X_train)
```

### Wrapper Methods
Train/evaluate a model on different feature subsets to find the best combo. Accurate but expensive.

```python
from sklearn.feature_selection import RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
rfe.fit(X_train, y_train)
```

### Embedded Methods
Feature selection happens **during** model training itself.
- Lasso: L1 penalty zeroes out coefficients
- Tree-based models: `feature_importances_`

### PCA (Principal Component Analysis)
Unsupervised dimensionality reduction — transforms correlated features into uncorrelated "principal components" ordered by variance explained. **Must scale data first.**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=0.95)  # keep 95% variance
X_pca = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)
```
❌ Components lose direct interpretability (they're linear combos of originals).

### Curse of Dimensionality
As features increase, data becomes sparse and distance-based methods (KNN, K-Means) lose meaning — "nearest neighbor" stops being meaningful. Mitigate with PCA, feature selection, or more data.
# Part 6 — Data Preprocessing

### Missing Values
```python
# Check
df.isnull().sum()

# Drop
df.dropna(subset=['col'], inplace=True)

# Impute
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')  # or 'mean', 'most_frequent'
X_imputed = imputer.fit_transform(X)
```

### Outliers
Detected via Z-score or IQR. Decide: remove, cap (clip), transform (log), or use robust model.

```python
Q1, Q3 = df['col'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df_clean = df[(df['col'] >= lower) & (df['col'] <= upper)]
```

### Feature Scaling
Required for distance-based (KNN, SVM, K-Means) and gradient-based (linear/logistic regression, neural nets) models. **Not needed** for tree-based models.

### Standardization (Z-score)
`(x - mean) / std` → mean 0, std 1. Robust to outliers vs. Normalization. Preferred for PCA, linear/logistic regression, SVM.

```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X_train)
```

### Normalization (Min-Max)
`(x - min) / (max - min)` → bounded [0, 1]. Sensitive to outliers. Common for neural net inputs, image pixels.

```python
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(X_train)
```

### Label Encoding
Assigns integers to categories. Use for **ordinal** data (has natural order: Low/Medium/High).

```python
from sklearn.preprocessing import LabelEncoder
df['size_encoded'] = LabelEncoder().fit_transform(df['size'])  # Small=0, Medium=1, Large=2
```

### One-Hot Encoding
Creates a binary column per category. Use for **nominal** data (no order: colors, cities) — avoids implying false order.

```python
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)
```

### Class Imbalance
One class dominates → model biased toward majority, high accuracy but useless minority-class recall. Fix with: resampling, class weights, or better metrics (F1, not accuracy).

```python
model = LogisticRegression(class_weight='balanced')
```

### SMOTE
Generates *synthetic* minority samples by interpolating between existing minority points and their neighbors (not simple duplication).

```python
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
```

### ADASYN
Like SMOTE, but generates **more** synthetic samples for minority points near the decision boundary (harder cases).

```python
from imblearn.over_sampling import ADASYN
X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)
```
# Part 7 — Model Evaluation

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, preds)
#            Pred 0   Pred 1
# Actual 0     TN       FP
# Actual 1     FN       TP
```

### Accuracy
`(TP + TN) / Total` — misleading on imbalanced data.

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, preds)
```

### Precision
`TP / (TP + FP)` — of predicted positives, how many were correct. Matters when **false positives are costly** (e.g., spam filter blocking real emails).

### Recall
`TP / (TP + FN)` — of actual positives, how many were caught. Matters when **false negatives are costly** (e.g., missing a disease/fraud case).

### F1 Score
Harmonic mean of Precision & Recall: `2 * (P*R)/(P+R)`. Good single metric for imbalanced data.

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, preds))  # gives precision, recall, f1 per class
```

### ROC Curve & ROC-AUC
ROC plots TPR vs FPR across all thresholds. AUC = area under that curve (1.0 = perfect, 0.5 = random guessing).

```python
from sklearn.metrics import roc_auc_score, roc_curve
auc = roc_auc_score(y_test, probs)
fpr, tpr, thresholds = roc_curve(y_test, probs)
```
Note: on heavily imbalanced data, prefer **Precision-Recall AUC** over ROC-AUC (ROC-AUC can look deceptively good).

### Regression Metrics — Quick Recall
MAE, MSE, RMSE, R² (full detail in Part 2).

### Classification Metrics — Quick Recall
Confusion Matrix → Accuracy, Precision, Recall, F1, ROC-AUC. Pick based on class balance and cost of FP vs FN — never rely on accuracy alone for imbalanced problems.
# Part 8 — Explainable AI

### Model Interpretability
How well a human can understand *why* a model made a prediction. Linear models/trees: interpretable by default. Neural nets/large ensembles: black boxes, need extra tools (SHAP/LIME).

### SHAP
Game-theory based (Shapley values) — fairly attributes a prediction's outcome across all features. Gives both global (overall importance) and local (single-prediction) explanations. Theoretically rigorous but computationally heavier.

```python
import shap
explainer = shap.TreeExplainer(model)   # for tree-based models
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)   # global
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])  # local
```

### LIME
Explains a **single prediction** by perturbing inputs around it and fitting a simple local linear model to approximate the black box locally. Faster than SHAP, less theoretically grounded.

```python
from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['0','1'])
exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
exp.show_in_notebook()
```

### Global vs Local Explanations
- **Global**: overall feature importance across the whole dataset (e.g., "income matters most on average")
- **Local**: why *this one* prediction happened (e.g., "this applicant was rejected mainly due to debt ratio")
- SHAP can do both; LIME is local-only.
# Part 9 — Hyperparameter Optimization

### Hyperparameters vs Parameters
- **Hyperparameters**: set *before* training, control how learning happens (e.g., `learning_rate`, `max_depth`, `n_estimators`)
- **Parameters**: learned automatically *from* data during training (e.g., weights, coefficients)

### Hyperparameter Tuning
Systematic search for the best hyperparameter combination using validation/cross-validation performance.

### Grid Search
Exhaustively tries every combination in a specified grid. Thorough but slow (combinatorial explosion).

```python
from sklearn.model_selection import GridSearchCV
params = {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3]}
grid = GridSearchCV(xgb.XGBClassifier(), params, cv=5, scoring='f1')
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

### Random Search
Samples a fixed number of random combinations instead of all of them. Often nearly as good, much faster.

```python
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(xgb.XGBClassifier(), params, n_iter=20, cv=5, scoring='f1', random_state=42)
random_search.fit(X_train, y_train)
```

### Cross Validation with Grid Search
Each hyperparameter combo is evaluated via K-Fold CV (not a single split) → more robust "best" choice. This is exactly what `GridSearchCV`/`RandomizedSearchCV` do internally (`cv=5` parameter above).
# Part 10 — Deep Learning Basics

### Neural Networks
Layers of connected neurons. Each neuron = weighted sum of inputs + bias, passed through an activation function. Input layer → hidden layer(s) → output layer. "Deep" = multiple hidden layers.

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 1), nn.Sigmoid()
)
```

### Activation Functions
Introduce non-linearity — without them, stacking layers is just one big linear function.

| Function | Formula | Use |
|---|---|---|
| Sigmoid | `1/(1+e^-x)` | Binary output (0-1) |
| Tanh | `(e^x - e^-x)/(e^x + e^-x)` | Hidden layers, zero-centered |
| ReLU | `max(0, x)` | Default for hidden layers, cheap, fixes vanishing gradient (mostly) |
| Softmax | `e^xi / Σe^xj` | Multi-class output (probabilities sum to 1) |

### Forward Propagation
Input flows through the network layer by layer → produces a prediction. `output = activation(W.x + b)` repeated per layer.

### Backpropagation
Computes how much each weight contributed to the error (via chain rule), propagating the error backward from output to input layer, producing gradients for each weight.

```python
loss = criterion(model(X_batch), y_batch)
loss.backward()      # backpropagation — computes gradients
optimizer.step()     # updates weights using those gradients
optimizer.zero_grad()
```

### Vanishing Gradient Problem
**What it is:** In deep networks, gradients are computed via the chain rule — multiplying many small numbers together (from derivatives of activations like Sigmoid/Tanh, which are <1). As the error signal propagates backward through many layers, the gradient shrinks exponentially, becoming near-zero by the time it reaches early layers. Result: early layers barely update — the network stops learning, especially in early layers.

**Why it happens:** Sigmoid's derivative max is 0.25; Tanh's max is 1.0 but still <1 almost everywhere. Stack 10+ layers → gradient = product of 10+ numbers each <1 → vanishes toward 0.

**Symptoms:** Training loss plateaus early, deep layers near the input barely change weights during training, very slow/stalled convergence.

**Fixes:**
- Use **ReLU** (and variants: Leaky ReLU, ELU) instead of Sigmoid/Tanh in hidden layers — derivative is 1 for positive inputs, doesn't shrink
- **Batch Normalization** — normalizes layer inputs, keeps gradients in a healthy range
- **Residual connections / skip connections** (ResNets) — let gradients flow directly through shortcut paths
- **Proper weight initialization** (He init for ReLU, Xavier/Glorot for Tanh)
- **Gradient clipping** — caps gradient magnitude (more common for the *opposite* problem, exploding gradients, but related)

```python
# He initialization example (good with ReLU)
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

# Batch norm in a layer
nn.Sequential(
    nn.Linear(64, 64),
    nn.BatchNorm1d(64),
    nn.ReLU()
)
```

**Related — Exploding Gradient Problem:** the opposite issue — gradients grow exponentially large through layers, causing unstable updates/NaN losses. Fixed with gradient clipping (`torch.nn.utils.clip_grad_norm_`) or smaller learning rates.

### Optimizers
Use gradients (from backprop) to actually update weights.

| Optimizer | Idea |
|---|---|
| SGD | Fixed-size step in gradient direction |
| Momentum | Adds "velocity" from past gradients to smooth updates |
| RMSProp | Adapts learning rate per-parameter using recent gradient magnitude |
| Adam | Combines Momentum + RMSProp — most common modern default |

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### Fine-Tuning
Take a pretrained model, continue training it (usually with a small learning rate) on your specific, smaller dataset.

```python
for param in model.base_layers.parameters():
    param.requires_grad = False   # freeze pretrained backbone
# only train the new task-specific head
optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-4)
```

### Transfer Learning
Broader concept: reuse knowledge from a model trained on one task/domain for a different but related task. Fine-tuning is one way to do this; freezing layers and only training a new head is another.
# Part 11 — Complete ML Pipeline

```
Data Collection → Data Cleaning → EDA → Feature Engineering → Feature Selection
→ Train-Test Split → Cross Validation → Model Selection → Hyperparameter Tuning
→ Model Evaluation → Interpretability → Deployment → Monitoring
```

| Stage | What happens |
|---|---|
| Data Collection | Pull raw data from DB/API/logs/scraping |
| Data Cleaning | Fix missing values, duplicates, outliers |
| EDA | Visualize distributions, correlations, class balance |
| Feature Engineering | Create new useful features |
| Feature Selection | Drop irrelevant/redundant features |
| Train-Test Split | Hold out untouched test set |
| Cross Validation | K-Fold on training data for reliable model comparison |
| Model Selection | Pick best algorithm based on CV results |
| Hyperparameter Tuning | GridSearchCV / RandomizedSearchCV |
| Model Evaluation | Final metrics on test set (Precision/Recall/F1/AUC) |
| Interpretability | SHAP/LIME to explain predictions to stakeholders |
| Deployment | Wrap model in an API (e.g., FastAPI) for production use |
| Monitoring | Track data drift, performance decay, trigger retraining |

```python
# Minimal example tying a few stages together
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', xgb.XGBClassifier())
])

params = {'model__max_depth': [3, 5], 'model__learning_rate': [0.05, 0.1]}
grid = GridSearchCV(pipe, params, cv=5, scoring='f1')
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Test F1:", f1_score(y_test, grid.predict(X_test)))
```

In practice this pipeline is **iterative**, not strictly linear — evaluation often sends you back to feature engineering or data cleaning.
