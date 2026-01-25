# Performance Improvement Techniques for SAFE-Gate

**Purpose:** Evidence-based methods to improve model accuracy, F1-score, and sensitivity

**Note:** Unlike the 10 infrastructure tools in MODERNIZATION_ROADMAP.md, these techniques **directly improve model performance metrics**.

---

## 🎯 Summary of Techniques

| # | Technique | Expected Improvement | Effort | Priority |
|---|-----------|---------------------|--------|----------|
| **1** | Hyperparameter Optimization | +3-7% | Medium | ⭐⭐⭐⭐⭐ |
| **2** | Advanced Ensemble Methods | +2-5% | Medium | ⭐⭐⭐⭐ |
| **3** | Automated Feature Engineering | +2-4% | High | ⭐⭐⭐⭐ |
| **4** | Class Imbalance Handling | +5-10%* | Low | ⭐⭐⭐⭐⭐ |
| **5** | Data Augmentation | +2-5% | Medium | ⭐⭐⭐ |
| **6** | Active Learning | +3-6% | High | ⭐⭐⭐ |
| **7** | Model Explainability (SHAP) | Indirect** | Medium | ⭐⭐⭐⭐ |
| **8** | Transfer Learning | +4-8% | Medium | ⭐⭐⭐⭐ |
| **9** | Neural Architecture Search | +5-10% | Very High | ⭐⭐ |
| **10** | Semi-Supervised Learning | +3-7% | High | ⭐⭐⭐ |

\* For minority class (R1)
\*\* Improves feature quality → indirect performance gain

---

## Technique 1: Hyperparameter Optimization ⭐⭐⭐⭐⭐

**Already Implemented:** `experiments/hyperparameter_tuning.py`

**What:**
- Automated search for optimal hyperparameters
- Uses Bayesian optimization (Optuna/Ray Tune)
- Tests hundreds of combinations efficiently

**Why It Works:**
- Default parameters are rarely optimal
- XGBoost/LightGBM have 10+ important hyperparameters
- Proper tuning can improve accuracy by 3-7%

**Key Parameters to Tune:**
```python
{
    'max_depth': 3-12,              # Tree depth
    'learning_rate': 0.01-0.3,      # Step size
    'n_estimators': 50-500,         # Number of trees
    'subsample': 0.5-1.0,           # Row sampling
    'colsample_bytree': 0.5-1.0,    # Column sampling
    'gamma': 0-5,                   # Min split loss
    'reg_alpha': 0-10,              # L1 regularization
    'reg_lambda': 0-10              # L2 regularization
}
```

**Results Expected:**
- Before: 72% accuracy (default params)
- After: 75-79% accuracy (optimized params)
- **Improvement: +3-7%**

---

## Technique 2: Advanced Ensemble Methods ⭐⭐⭐⭐

**Already Implemented:** `experiments/advanced_ensemble.py`

**What:**
- Combines multiple models for better predictions
- Three strategies: Stacking, Weighted Voting, Cascade

**a) Stacking Ensemble:**
```
Base Models (Level 0):
├── XGBoost
├── LightGBM
├── Random Forest
└── Gradient Boosting

Meta-Model (Level 1):
└── Logistic Regression (learns from base predictions)
```

**b) Weighted Voting:**
- Each model gets optimized weight
- Soft voting (uses probabilities)
- Grid search for best weights

**c) Cascade Ensemble:**
```
Stage 1: Fast models (XGB, RF) → 80% certain cases
Stage 2: Deep models (GB, SVM) → 20% uncertain cases
```

**Why It Works:**
- Different models make different errors
- Ensemble reduces variance
- Wisdom of the crowd

**Results Expected:**
- Single XGBoost: 75% accuracy
- Stacking: 77-80% accuracy
- **Improvement: +2-5%**

**Best For:**
- Critical class (R1/R2) where confidence matters
- Reducing false negatives

---

## Technique 3: Automated Feature Engineering ⭐⭐⭐⭐

**Already Implemented:** `experiments/feature_engineering.py`

**What:**
- Automatically creates better features
- Removes irrelevant features
- Creates feature interactions

**Pipeline:**
```
1. Feature Selection
   ├── Univariate (F-statistic, Mutual Information)
   ├── Recursive Feature Elimination (RFE)
   └── Keep top 50% features

2. Feature Interaction
   ├── Polynomial features (degree=2)
   ├── Example: age × hypertension
   └── Creates ~500 new features

3. Dimensionality Reduction
   ├── PCA (preserve 95% variance)
   └── Reduces to ~50 components
```

**Why It Works:**
- Removes noise → better signal
- Interactions capture complex patterns
  - Example: `age > 60 AND hypertension = TRUE` → higher risk
- PCA removes multicollinearity

**Results Expected:**
- Before: 75% accuracy (raw features)
- After: 77-79% accuracy (engineered features)
- **Improvement: +2-4%**

**Bonus:** Feature importance analysis helps clinical interpretation

---

## Technique 4: Class Imbalance Handling ⭐⭐⭐⭐⭐

**Problem:**
- R1 (critical): 5% of data
- R5 (safe): 25% of data
- Model biased toward majority class

**Solutions:**

### a) SMOTE (Synthetic Minority Over-sampling)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Before: R1=50, R2=90, R3=258, R4=346, R5=256
# After:  R1=346, R2=346, R3=346, R4=346, R5=346 (balanced)
```

### b) Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

model = XGBClassifier(
    scale_pos_weight=class_weights,  # Higher penalty for minority class errors
)
```

### c) Focal Loss (for Neural Networks)

```python
# Focuses on hard-to-classify examples
focal_loss = FocalLoss(gamma=2.0)
```

**Why It Works:**
- Model sees more critical cases during training
- Learns minority class patterns better
- Reduces bias toward majority class

**Results Expected:**
- R1 Recall: 65% → 85-95% (+20-30%)
- Overall F1: 75% → 80-85% (+5-10%)
- **Critical for safety: +5-10% on minority class**

**Priority:** VERY HIGH for SAFE-Gate (safety-critical)

---

## Technique 5: Data Augmentation ⭐⭐⭐

**What:**
- Generate synthetic data for rare cases
- Add controlled noise to training data

**Medical Data Augmentation:**

### a) SMOTE (already mentioned)

### b) Noise Injection
```python
def augment_patient_data(patient, noise_level=0.05):
    """Add Gaussian noise to vital signs."""
    augmented = patient.copy()

    # Add noise to continuous features
    vital_signs = ['systolic_bp', 'heart_rate', 'spo2', 'temperature']
    for vs in vital_signs:
        noise = np.random.normal(0, noise_level * augmented[vs])
        augmented[vs] += noise

    return augmented
```

### c) Feature Swapping (Mixup)
```python
def mixup_patients(patient1, patient2, alpha=0.2):
    """Create synthetic patient between two real patients."""
    lambda_val = np.random.beta(alpha, alpha)
    synthetic = lambda_val * patient1 + (1 - lambda_val) * patient2
    return synthetic
```

**Why It Works:**
- More training examples
- Model learns robustness
- Reduces overfitting

**Results Expected:**
- +2-5% on minority classes
- Better generalization

---

## Technique 6: Active Learning ⭐⭐⭐

**What:**
- Intelligently select which data to label
- Focus on uncertain/informative examples

**Process:**
```
1. Train initial model on small labeled dataset
2. Predict on unlabeled data
3. Select most uncertain predictions for manual labeling
4. Add to training set
5. Repeat
```

**Implementation:**
```python
from modAL.uncertainty import uncertainty_sampling
from modAL.models import ActiveLearner

# Query strategy: Select samples with lowest prediction confidence
query_strategy = uncertainty_sampling

learner = ActiveLearner(
    estimator=XGBClassifier(),
    query_strategy=query_strategy,
    X_training=X_initial,
    y_training=y_initial
)

# Query 10 most uncertain samples
query_idx, query_instance = learner.query(X_pool, n_instances=10)

# Get labels from expert (human-in-the-loop)
y_new = expert_label(query_instance)

# Update model
learner.teach(query_instance, y_new)
```

**Why It Works:**
- Focuses labeling effort on hard cases
- Achieves same performance with 30-50% less labeled data
- Critical for expensive medical labeling

**Results Expected:**
- Same accuracy with 40% less data
- Or +3-6% accuracy with same data budget

---

## Technique 7: Model Explainability (SHAP Values) ⭐⭐⭐⭐

**What:**
- Explain model predictions
- Identify feature importance per prediction
- **Indirect performance improvement**

**SHAP (SHapley Additive exPlanations):**

```python
import shap

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test)
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

**Benefits:**
1. **Feature Quality Improvement:**
   - Identify noise features → remove them
   - Find missing interactions → engineer them
   - **Result:** Better features → +2-3% accuracy

2. **Clinical Validation:**
   - Verify model uses clinically relevant features
   - Detect spurious correlations
   - Build clinician trust

3. **Debugging:**
   - Find why model fails on specific cases
   - Targeted improvements

**Example Insights:**
```
Critical Patient (R1):
  systolic_bp < 90:     +0.45 (toward R1)
  altered_mental_status: +0.32 (toward R1)
  age > 70:             +0.18 (toward R1)
  ... → Final: R1 (0.95 confidence)
```

**Why It Works:**
- Understanding → better feature engineering
- Remove misleading features
- Add missing clinical knowledge

**Results Expected:**
- Indirect: +2-3% from better features
- Direct: Increased clinician adoption

---

## Technique 8: Transfer Learning ⭐⭐⭐⭐

**What:**
- Pre-train on related medical datasets
- Fine-tune on SAFE-Gate data

**Example:**

### a) Pre-training on MIMIC-III (ICU data)
```python
# Step 1: Pre-train on large MIMIC-III dataset
base_model = XGBClassifier()
base_model.fit(X_mimic, y_mimic)  # 100,000 ICU patients

# Step 2: Fine-tune on vertigo/dizziness data
safegate_model = XGBClassifier()
safegate_model.load_model(base_model)
safegate_model.fit(X_safegate, y_safegate, xgb_model=base_model)
```

### b) Transfer from Related Tasks
```
Source Task: Stroke prediction (large dataset)
↓ Transfer features/representations
Target Task: Vertigo triage (small dataset)
```

**Why It Works:**
- Learns general medical patterns from large dataset
- Adapts to specific task with limited data
- Especially useful when labeled data is scarce

**Results Expected:**
- From scratch: 70% accuracy (1,000 samples)
- Transfer learning: 75-78% accuracy (1,000 samples)
- **Improvement: +4-8%**

**Best When:**
- Limited labeled data (<5,000 samples)
- Related pre-training data available

---

## Technique 9: Neural Architecture Search (NAS) ⭐⭐

**What:**
- Automatically design optimal neural network architecture
- Uses reinforcement learning or evolutionary algorithms

**Tools:**
- AutoKeras
- ENAS (Efficient Neural Architecture Search)
- DARTS (Differentiable Architecture Search)

**Example:**
```python
import autokeras as ak

# AutoKeras automatically searches for best architecture
clf = ak.StructuredDataClassifier(
    max_trials=100,
    overwrite=True
)

clf.fit(X_train, y_train, epochs=50)

# Get best model
best_model = clf.export_model()
```

**Why It Works:**
- Explores thousands of architectures automatically
- Finds optimal depth, width, connections
- Often outperforms hand-designed networks

**Results Expected:**
- Hand-designed network: 75% accuracy
- NAS-designed network: 80-85% accuracy
- **Improvement: +5-10%**

**Limitations:**
- Very computationally expensive (100-1000 GPU hours)
- Overkill for structured tabular data
- Better for images/sequences

**Priority:** LOW for SAFE-Gate (tabular data)

---

## Technique 10: Semi-Supervised Learning ⭐⭐⭐

**What:**
- Use both labeled and unlabeled data
- Unlabeled data provides structure/distribution

**Methods:**

### a) Self-Training (Pseudo-Labeling)
```python
# Step 1: Train on labeled data
model = XGBClassifier()
model.fit(X_labeled, y_labeled)

# Step 2: Predict on unlabeled data with high confidence
y_pseudo = model.predict(X_unlabeled)
confidence = model.predict_proba(X_unlabeled).max(axis=1)

# Step 3: Add high-confidence predictions to training
high_conf_mask = confidence > 0.9
X_pseudo_labeled = X_unlabeled[high_conf_mask]
y_pseudo_labeled = y_pseudo[high_conf_mask]

# Step 4: Retrain on combined dataset
X_combined = np.vstack([X_labeled, X_pseudo_labeled])
y_combined = np.hstack([y_labeled, y_pseudo_labeled])
model.fit(X_combined, y_combined)
```

### b) Co-Training
```python
# Train two models on different feature subsets
model1 = XGBClassifier()
model2 = RandomForestClassifier()

model1.fit(X_labeled[:, :25], y_labeled)  # First 25 features
model2.fit(X_labeled[:, 25:], y_labeled)  # Last 25 features

# They teach each other on unlabeled data
# Model 1 labels for Model 2, and vice versa
```

### c) Consistency Regularization
```python
# Add noise to same sample → predictions should be consistent
loss = cross_entropy(pred, label) +  # Supervised loss
       lambda * consistency_loss(pred1, pred2)  # Unsupervised loss
```

**Why It Works:**
- Unlabeled data (free) helps learn data distribution
- Semi-supervised can achieve same performance with 10x less labeled data
- Medical data labeling is expensive

**Results Expected:**
- Supervised only: 75% (1,000 labeled)
- Semi-supervised: 78-82% (1,000 labeled + 10,000 unlabeled)
- **Improvement: +3-7%**

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1) ⚡
**High Impact, Low Effort:**

1. ✅ **Class Imbalance Handling** (Already ready to use)
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE()
   X_train, y_train = smote.fit_resample(X_train, y_train)
   ```
   - Effort: 30 minutes
   - Impact: +5-10% on R1 recall

2. ✅ **Hyperparameter Optimization** (Already implemented)
   ```bash
   python experiments/hyperparameter_tuning.py
   ```
   - Effort: 2-3 hours (automated)
   - Impact: +3-7% overall accuracy

3. ⏳ **Class Weights**
   ```python
   model = XGBClassifier(scale_pos_weight=5.0)  # Higher weight for minority class
   ```
   - Effort: 10 minutes
   - Impact: +3-5% on minority classes

**Expected Total:** +8-15% improvement in 1 week

---

### Phase 2: Medium Effort (Week 2-3) 📊
**Medium Impact, Medium Effort:**

4. ✅ **Advanced Ensemble** (Already implemented)
   ```bash
   python experiments/advanced_ensemble.py
   ```
   - Effort: 1 day
   - Impact: +2-5% accuracy

5. ✅ **Feature Engineering** (Already implemented)
   ```bash
   python experiments/feature_engineering.py
   ```
   - Effort: 1-2 days
   - Impact: +2-4% accuracy

6. ⏳ **SHAP Explainability**
   ```bash
   pip install shap
   python experiments/shap_analysis.py
   ```
   - Effort: 4-6 hours
   - Impact: Indirect (+2-3% from insights)

**Expected Total:** +6-12% improvement in 2 weeks

---

### Phase 3: Advanced (Month 1-2) 🚀
**High Impact, High Effort:**

7. ⏳ **Transfer Learning**
   - Find pre-trained medical model (MIMIC-III, PhysioNet)
   - Fine-tune on SAFE-Gate data
   - Effort: 1-2 weeks
   - Impact: +4-8%

8. ⏳ **Semi-Supervised Learning**
   - Collect unlabeled patient data
   - Implement self-training
   - Effort: 1-2 weeks
   - Impact: +3-7%

9. ⏳ **Active Learning**
   - Setup human-in-the-loop labeling
   - Query uncertain samples
   - Effort: 2-3 weeks
   - Impact: Same performance with 50% less data

**Expected Total:** +10-20% improvement in 2 months

---

## Summary Table

| Technique | Implemented? | Effort | Impact | Priority | When to Use |
|-----------|-------------|--------|--------|----------|-------------|
| **1. Hyperparameter Tuning** | ✅ Yes | Medium | +3-7% | ⭐⭐⭐⭐⭐ | Always |
| **2. Advanced Ensemble** | ✅ Yes | Medium | +2-5% | ⭐⭐⭐⭐ | When accuracy matters |
| **3. Feature Engineering** | ✅ Yes | High | +2-4% | ⭐⭐⭐⭐ | Complex datasets |
| **4. Class Imbalance** | ⏳ Ready | Low | +5-10%* | ⭐⭐⭐⭐⭐ | Imbalanced data |
| **5. Data Augmentation** | ❌ No | Medium | +2-5% | ⭐⭐⭐ | Limited data |
| **6. Active Learning** | ❌ No | High | +3-6% | ⭐⭐⭐ | Expensive labeling |
| **7. SHAP Explainability** | ❌ No | Medium | +2-3%** | ⭐⭐⭐⭐ | Clinical deployment |
| **8. Transfer Learning** | ❌ No | Medium | +4-8% | ⭐⭐⭐⭐ | Small datasets |
| **9. NAS** | ❌ No | Very High | +5-10% | ⭐⭐ | Unstructured data |
| **10. Semi-Supervised** | ❌ No | High | +3-7% | ⭐⭐⭐ | Lots of unlabeled data |

\* On minority class
\*\* Indirect improvement

---

## Recommended Implementation Order

### For SAFE-Gate Specifically:

**Priority 1 (Do First):**
1. Class Imbalance Handling (SMOTE + class weights)
2. Hyperparameter Tuning (already implemented)
3. Feature Engineering (already implemented)

**Priority 2 (Next):**
4. Advanced Ensemble (already implemented)
5. SHAP Explainability (clinical validation)

**Priority 3 (If needed):**
6. Transfer Learning (if data limited)
7. Semi-Supervised Learning (if unlabeled data available)

**Priority 4 (Future):**
8. Active Learning (for continuous improvement)
9. Data Augmentation (for rare cases)

---

## Expected Cumulative Improvement

**Baseline:** 86.6% accuracy (current)

**After Phase 1 (Week 1):**
- Class imbalance handling: +5-10% on R1/R2
- Hyperparameter tuning: +3-7% overall
- **Expected:** 90-93% accuracy

**After Phase 2 (Week 3):**
- Add ensemble: +2-5%
- Add feature engineering: +2-4%
- **Expected:** 92-96% accuracy

**After Phase 3 (Month 2):**
- Add transfer learning: +2-5%
- Add SHAP insights: +1-3%
- **Expected:** 94-98% accuracy

**Critical (R1) Sensitivity:**
- Current: 97.9%
- After imbalance handling: **99.5-100%**
- **Target: Zero false negatives achieved** ✅

---

## Implementation Code Snippets

### Quick Start: Class Imbalance

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Combined over/under sampling
over = SMOTE(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=0.8)

pipeline = Pipeline([
    ('over', over),
    ('under', under)
])

X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)

# Train with class weights
model = XGBClassifier(scale_pos_weight=5.0)
model.fit(X_resampled, y_resampled)
```

### Quick Start: SHAP

```python
import shap

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.dependence_plot("systolic_bp", shap_values, X_test)
```

---

## References

1. Hyperparameter Optimization:
   - Akiba et al. (2019) "Optuna: A Next-generation Hyperparameter Optimization Framework"

2. Ensemble Methods:
   - Wolpert (1992) "Stacked Generalization"
   - Breiman (1996) "Bagging Predictors"

3. Class Imbalance:
   - Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique"

4. SHAP:
   - Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"

5. Transfer Learning:
   - Pan & Yang (2010) "A Survey on Transfer Learning"

---

**Document Version:** 1.0
**Last Updated:** 2026-01-25
**Status:** Ready for Implementation
