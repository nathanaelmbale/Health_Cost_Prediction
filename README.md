
# Healthcare Cost Prediction Using Neural Networks

## Quick Overview 

**Problem:** Predict individual healthcare costs using regression analysis on demographic and health data, achieving Mean Absolute Error (MAE) under **$3,500**.

**Solution:** Built a deep neural network regression model with feature normalization and categorical encoding to predict medical expenses from patient characteristics.

**Impact:** Achieved **$2,800 MAE** (20% better than the requirement), demonstrating proficiency in regression modeling, data preprocessing, and TensorFlow/Keras implementation.



## Step-by-Step Implementation

### Step 1: Environment Setup & Data Loading

**What:** Imported TensorFlow 2.x, Keras, and a healthcare insurance dataset.

**Dataset:** `insurance.csv` containing patient information and medical costs.

**Key Libraries:**
- TensorFlow / Keras for neural networks
- Pandas for data manipulation
- Matplotlib for visualization
- `tensorflow_docs` for training callbacks

---

### Step 2: Data Exploration

**What:** Examined dataset structure and features.

**Features Identified:**
- **Numerical:** age, bmi, children
- **Categorical:** sex, smoker, region
- **Target Variable:** expenses (healthcare costs)

**Initial Analysis:** Used `dataset.tail()` to inspect data structure and identify preprocessing needs.

---

### Step 3: Categorical Data Encoding

**Problem:** Neural networks require numerical inputs, but the dataset contains text categories.

**Solution:** Manual mapping of categorical variables to numerical values.

- Sex: `male → 0`, `female → 1`
- Smoker: `no → 0`, `yes → 1`
- Region:
  - southwest → 0
  - southeast → 1
  - northwest → 2
  - northeast → 3

**Key Action:** Applied string stripping to handle whitespace before mapping.

**Impact:** Converted all features to numerical format for model compatibility.

---

### Step 4: Train-Test Split

**Split Ratio:** 80% training, 20% testing.

**Method:** Random sampling using `frac=0.2` for the test set.

**Implementation:**
- Test dataset: 20% random sample from original data
- Train dataset: Remaining 80%, created by dropping test indices

**Label Separation:**
- Popped the `expenses` column to create labels
- `train_labels`: target values for training
- `test_labels`: target values for evaluation

---

### Step 5: Feature Normalization

**Problem:** Features exist on different scales (age: 18–64, BMI: 15–50, expenses: $1k–$60k), causing training instability.

**Solution:** Z-score normalization (standardization).

- Calculated mean and standard deviation from **training data only**
- Applied formula: `(x − mean) / std`

**Critical Decision:** Used training statistics for both training and test data to prevent data leakage.

**Impact:**
- Faster convergence during training
- Improved gradient descent stability
- Better model generalization

---

### Step 6: Neural Network Architecture Design

**Model Type:** Sequential feedforward neural network for regression.

**Architecture:**
- Input layer: 6 features after encoding
- Hidden Layer 1: 64 neurons, ReLU activation
- Hidden Layer 2: 64 neurons, ReLU activation
- Output Layer: 1 neuron for continuous cost prediction

**Rationale:**
- Two hidden layers capture non-linear relationships
- ReLU activation avoids vanishing gradients
- 64 neurons balance learning capacity and overfitting risk

---

### Step 7: Model Compilation

**Optimizer:** RMSprop (learning rate = 0.001)
- Adaptive learning rate suitable for regression
- Handles sparse gradients effectively

**Loss Function:** Mean Squared Error (MSE)
- Standard loss for regression
- Penalizes large errors more heavily

**Metrics Tracked:**
- MAE (Mean Absolute Error): primary evaluation metric
- MSE (Mean Squared Error): secondary monitoring metric

---

### Step 8: Model Training

**Training Configuration:**
- Epochs: 500
- Validation split: 20% of training data
- Verbose: 0 with `EpochDots` callback

**Data Used:** Normalized training data

**Monitoring:** Validation loss tracked to detect overfitting.

**Result:** Model learned relationships between patient features and healthcare costs.

---

### Step 9: Model Evaluation

**Test Evaluation:** `model.evaluate()` on normalized test data.

**Key Metric:** Mean Absolute Error (MAE)

**Performance Achieved:** **$2,800 MAE**

**Target:** ≤ $3,500

✅ Beat target by **$700** (20% improvement).

**Interpretation:** Model predictions are, on average, $2,800 away from actual healthcare costs.

---

### Step 10: Prediction Visualization

**What:** Generated a scatter plot of true versus predicted expenses.

**Plot Elements:**
- X-axis: Actual healthcare costs (`test_labels`)
- Y-axis: Model predictions
- Diagonal line: Perfect prediction reference
- Equal aspect ratio for accurate comparison

**Interpretation:**
- Points close to the diagonal indicate accurate predictions
- Scatter distribution highlights model strengths and weaknesses
- Visual confirmation of $2,800 MAE performance

---

### Step 11: Final Validation

**Automated Test:** Built-in evaluation verifying MAE ≤ $3,500.

**Result:**

✅ *"You passed the challenge. Great job!"*

**Verification:** Model successfully generalizes to unseen test data.

