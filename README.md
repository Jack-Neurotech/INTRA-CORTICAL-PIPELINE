# Intracortical BCI Pipeline

# 1. What Is the Project and What Does It Do?

* **Project:** Intracortical BCI Neural Decoding Pipeline

* **Purpose:** A Python-based neuroinformatics and computational neuroscience pipeline for loading recorded intracortical neural activity, extracting behavioral information, processing neural population activity, and decoding recorded finger movement.

* **Input:** A recorded intracortical neurophysiology dataset stored in an NWB file.

* **Primary goal:** Convert recorded neural activity into numerical features that can be used to predict recorded behavioral movement.

* **The pipeline performs the following steps:**

  * Loads an NWB neurophysiology file.
  * Extracts the recorded population of neurons.
  * Determines the number of recorded neurons.
  * Determines the recording duration.
  * Extracts the trial table.
  * Identifies the recorded behavioral variables.
  * Selects finger velocity as the behavioral target.
  * Extracts the continuous finger-velocity measurements.
  * Converts neural activity into numerical population features.
  * Bins the neural activity into 20-ms time windows.
  * Temporally aligns the neural population activity with the behavioral measurements.
  * Removes invalid neural or behavioral measurements.
  * Creates the neural feature matrix used by the decoder.
  * Creates the behavioral target matrix used by the decoder.
  * Divides the aligned dataset into training and testing data.
  * Trains a Ridge regression decoder.
  * Uses neural population activity to predict finger velocity.
  * Predicts two movement dimensions:

    * X-axis finger velocity
    * Y-axis finger velocity
  * Evaluates the decoder predictions against the recorded behavioral measurements.
  * Calculates quantitative performance measurements.
  * Reports the decoder results.

* **Neural input:**

  * Recorded activity from 130 neurons.
  * Neural activity is represented as a population feature matrix.
  * The population feature matrix contained:

```text
32,455 × 130
```

* This represents:

```text
32,455 time points
130 neurons
```

* **Behavioral target:**

  * Recorded finger velocity.
  * The target contains two movement dimensions:

```text
X velocity
Y velocity
```

* The behavioral target contained:

```text
32,440 × 2
```

* **Temporal processing:**

  * Neural activity is represented using 20-ms bins.
  * Behavioral measurements are temporally aligned with the neural population activity.
  * The final aligned dataset contained:

```text
32,440 neural samples
32,440 behavioral samples
```

* **Decoder:**

  * The project uses Ridge regression.
  * Neural population activity is the input.
  * Finger velocity is the prediction target.

* **Decoder configuration:**

```python
test_size=0.20
alpha=1.0
```

* **Evaluation measurements:**

  * R²
  * RMSE
  * MAE
  * Pearson correlation

* **Results:**

  * X-axis finger velocity:

```text
R²:        0.1284795848
RMSE:      75.48333927
MAE:       43.40040125
Pearson:   0.3638689504
```

* Y-axis finger velocity:

```text
R²:        0.1062295295
RMSE:      58.70979109
MAE:       37.02067597
Pearson:   0.3310680174
```

* **Overall concept:**

  * The project takes recorded intracortical neural activity.
  * Python extracts the neural and behavioral information.
  * Neural activity is converted into numerical population features.
  * Neural and behavioral measurements are temporally aligned.
  * The neural population becomes the input to a machine-learning decoder.
  * Finger velocity becomes the target.
  * The Ridge decoder learns the relationship between neural activity and movement.
  * The trained decoder predicts finger velocity from neural activity.
  * Statistical measurements quantify the relationship between predicted and recorded movement.

* **In simple terms:**

  * **Recorded neural activity → numerical neural features → temporal alignment → Ridge decoder → predicted finger movement → statistical evaluation**

* **What the project demonstrates:**

  * Python programming
  * Scientific computing
  * Numerical data processing
  * Computational neuroscience
  * Neuroinformatics
  * Intracortical neural data analysis
  * Neural population analysis
  * Temporal data alignment
  * Machine learning
  * Neural decoding
  * Regression
  * Quantitative evaluation

# 2. Python Principles, General Structure, Packages, and Associated Principles

## A. Python Principles and the General Structure

The intracortical BCI pipeline is built from fundamental Python programming concepts. These concepts provide the structure that allows the scientific packages, neural data, behavioral data, and machine-learning decoder to work together.

---

### Variables

* Variables store information so the program can use it later.
* The pipeline uses variables to store neural data, behavioral data, recording information, feature matrices, decoder settings, and results.

```python
bin_size = 0.020

behavior_target_name = "finger_vel"

number_of_neurons = (
    neural_data.get_number_of_units()
)
```

In these examples:

* `bin_size` stores the temporal bin size.
* `behavior_target_name` stores the name of the behavioral variable.
* `number_of_neurons` stores the number of recorded neurons.

---

### Data Types

Python can store different kinds of information.

Common types used in the pipeline include:

* **Numbers** — bin sizes, recording durations, statistical measurements, and decoder parameters.
* **Strings** — behavioral-variable names and file information.
* **Lists** — collections of behavioral objects or column names.
* **Arrays** — neural and behavioral measurements.
* **Objects** — organized structures containing data and functionality.
* **Dictionaries** — collections of named results such as decoder metrics.

Example:

```python
behavior_target_name = "finger_vel"

bin_size = 0.020
```

The first variable contains text, while the second contains a numerical value.

---

### Lists and Collections

Lists allow multiple pieces of information to be stored together.

For example, the trial table contains multiple column names:

```python
trial_columns = (
    behavior_data.get_trial_columns()
)
```

The program can then work with the collection of available trial variables.

---

### Indexing

Indexing allows Python to access a specific element inside a collection.

```python
aligned_times[0]
```

gets the first aligned time point.

```python
aligned_times[-1]
```

gets the final aligned time point.

Indexing is important when working with individual neural samples, behavioral measurements, and time points.

---

### Functions

Functions organize operations into reusable pieces of code.

```python
decoder.train()
```

A function can:

1. Receive input.
2. Process the input.
3. Return a result.

The general structure is:

**Input → Processing → Output**

Functions allow different parts of the intracortical pipeline to perform specific operations without placing every operation into one large block of code.

---

### Loops

Loops allow Python to repeat an operation.

```python
for name in behavior_objects:

    print(name)
```

The pipeline uses loops to work through collections of behavioral objects and other recorded information.

Instead of manually processing every item, Python repeats the same operation for each item.

---

### Conditional Statements

Conditional statements allow the program to make decisions.

```python
if not np.all(valid_mask):

    aligned_times = aligned_times[
        valid_mask
    ]
```

The condition determines whether invalid data need to be removed.

Conditions allow the pipeline to:

* Check whether data are valid.
* Check whether required information exists.
* Determine whether measurements should be retained.
* Prevent invalid values from entering the decoder.

---

### Boolean Logic

Boolean logic allows Python to evaluate conditions as `True` or `False`.

The pipeline combines neural and behavioral validity conditions:

```python
valid_mask = (
    np.all(
        np.isfinite(aligned_neural),
        axis=1
    )
    &
    np.all(
        np.isfinite(aligned_behavior),
        axis=1
    )
)
```

The `&` operator means both conditions must be satisfied.

Therefore, a sample is retained only when:

* The neural data are valid.
* The behavioral data are valid.

---

### Imports

Imports allow one Python module to access functionality created in another module.

```python
from nwb_reader import NWBReader
from neural_data import NeuralData
from behavior import BehaviorData
from features import FeatureExtractor
from alignment import TemporalAligner
from decoder import NeuralDecoder
```

The imports connect the main pipeline to the individual components that perform the scientific processing.

---

### Classes

A class is a blueprint for creating objects that contain related data and functions.

The project separates different stages of the pipeline into specialized classes.

```python
reader = NWBReader()

neural_data = NeuralData(nwbfile)

behavior_data = BehaviorData(nwbfile)

feature_extractor = FeatureExtractor(
    neural_data,
    behavior_data
)

aligner = TemporalAligner(...)

decoder = NeuralDecoder(...)
```

Each class has a specific responsibility.

---

### Objects

An object is a specific instance created from a class.

For example:

```python
decoder = NeuralDecoder(
    test_size=0.20,
    alpha=1.0
)
```

The `decoder` object contains the settings and methods needed to prepare, train, and evaluate the neural decoder.

---

### Methods

Methods are functions associated with an object.

For example:

```python
neural_data.extract_units()
```

```python
behavior_data.extract_trials()
```

```python
decoder.train()
```

```python
decoder.evaluate()
```

The method performs an operation using the information contained within the object.

---

### Arrays

Arrays are organized collections of numerical values.

Neural recordings contain large numbers of measurements, so the pipeline represents neural and behavioral data using numerical arrays.

For example:

```python
population_features
```

contains the neural population activity.

The population feature matrix was:

```text
32,455 × 130
```

meaning:

```text
32,455 time points
130 neurons
```

The behavioral target was:

```text
32,440 × 2
```

meaning:

```text
32,440 time points
2 movement dimensions
```

---

### Data Validation

Data validation checks whether information meets the requirements of the program before it is processed.

The pipeline checks the neural and behavioral data before training the decoder.

```python
valid_mask = (
    np.all(
        np.isfinite(aligned_neural),
        axis=1
    )
    &
    np.all(
        np.isfinite(aligned_behavior),
        axis=1
    )
)
```

This prevents `NaN` and infinite values from being passed into the machine-learning model.

---

### Object-Oriented Programming

Object-oriented programming organizes a program into objects that contain related data and operations.

The pipeline is divided into specialized objects:

```text
NWBReader
    ↓
NeuralData
    ↓
BehaviorData
    ↓
FeatureExtractor
    ↓
TemporalAligner
    ↓
NeuralDecoder
```

Each component performs a different part of the scientific analysis.

This makes the project easier to understand, modify, test, and reuse.

---

### Modular Programming

Modular programming divides a large program into smaller independent components.

The intracortical pipeline separates functionality into modules such as:

```text
nwb_reader.py
neural_data.py
behavior.py
features.py
alignment.py
decoder.py
main.py
```

Each module handles a particular part of the pipeline.

For example:

```python
from neural_data import NeuralData
```

allows the main program to use the neural-data module without rewriting its implementation.

---

### Data Flow

Data flow describes how information moves through the program.

The intracortical pipeline passes information through several processing stages:

```text
NWB file
   ↓
NWB object
   ↓
Neural data
   ↓
Population features
   ↓
Aligned neural data
   ↓
Behavioral target
   ↓
Ridge decoder
   ↓
Predictions
   ↓
Evaluation metrics
```

The Python code therefore acts as a sequence of transformations where the output from one stage becomes the input to the next.

---

### Exception Handling

Exceptions allow a program to identify conditions where it cannot safely continue.

For example, the behavioral-data component checks whether a trial table exists:

```python
if self.nwbfile.trials is None:

    raise ValueError(
        "The NWB file does not contain "
        "a trial table."
    )
```

This prevents missing information from silently entering the analysis.

---

### Numerical Computing

Numerical computing means performing mathematical operations on numerical data.

This is fundamental to the intracortical pipeline because neural recordings are numerical measurements.

NumPy is used for operations such as:

```python
np.asarray(...)
```

```python
np.isfinite(...)
```

and array-based calculations.

The numerical neural data ultimately become the input to the decoder.

---

### Machine-Learning Abstraction

Abstraction means using a higher-level object or function without manually implementing every underlying mathematical operation.

The pipeline does not manually implement all of the mathematics required for Ridge regression.

Instead, the decoder provides an interface for preparing, training, predicting, and evaluating the model.

```python
decoder.prepare_data(...)

decoder.train()

decoder.evaluate()

decoder.get_predictions()
```

This allows the project to focus on the neuroscience data-processing pipeline while the machine-learning implementation handles the underlying regression calculations.

## General Structure of the Intracortical BCI Pipeline

The fundamental structure of the pipeline is:

**1. Input**

* Locate and load the NWB neurophysiology file.
* Create an NWB data object.

**2. Neural Data Extraction**

* Extract the recorded neurons.
* Determine the number of neurons.
* Determine the recording duration.

**3. Behavioral Data Extraction**

* Extract the trial table.
* Identify behavioral variables.
* Select finger velocity as the behavioral target.

**4. Feature Extraction**

* Convert neural activity into numerical features.
* Create population-level neural features.
* Represent neural activity using 20-ms bins.

**5. Temporal Alignment**

* Match neural population activity with behavioral measurements.
* Create paired neural and behavioral observations.

**6. Data Validation**

* Check for invalid numerical values.
* Remove invalid observations.

**7. Decoder Preparation**

* Create the training dataset.
* Create the testing dataset.
* Separate neural inputs from behavioral targets.

**8. Machine Learning**

* Train a Ridge regression model.
* Learn the relationship between neural population activity and finger velocity.

**9. Prediction**

* Use neural activity to predict finger movement.
* Generate X and Y finger-velocity predictions.

**10. Evaluation**

* Compare predicted movement against recorded movement.
* Calculate R².
* Calculate RMSE.
* Calculate MAE.
* Calculate Pearson correlation.

The overall computational pattern is therefore:

**NWB File → Extract Neural Data → Extract Behavior → Create Features → Align Data → Validate Data → Train Decoder → Predict Movement → Evaluate Predictions**

# B. Packages and the Python Principles Associated With Them

## NumPy

**Purpose:**

* Numerical computing.
* Numerical array processing.
* Data validation.
* Mathematical operations on neural and behavioral data.

### Python principles associated with NumPy

* Variables
* Arrays
* Indexing
* Slicing
* Functions
* Mathematical operations
* Boolean logic
* Iteration

Example:

```python
np.asarray(...)
```

converts recorded data into numerical array structures that Python can process.

The pipeline also uses:

```python
np.isfinite(...)
```

to identify valid numerical values.

NumPy therefore provides the numerical foundation for processing the neural and behavioral datasets.

---

## PyNWB

**Purpose:**

* Working with NWB neurophysiology files.
* Reading structured neurophysiology recordings.
* Accessing neural units and behavioral information.

### Python principles associated with PyNWB

* Objects
* Attributes
* Methods
* Variables
* Functions
* Data structures

The NWB file is loaded into a Python object.

The pipeline can then access information contained within that object, including:

* Neural recordings
* Recorded units
* Trial information
* Behavioral measurements
* Recording metadata

The general relationship is:

**NWB File → Python NWB Object → Neural and Behavioral Data**

---

## scikit-learn

**Purpose:**

* Machine learning.
* Regression.
* Model training.
* Prediction.
* Model evaluation.

The intracortical pipeline uses Ridge regression to decode finger velocity from neural population activity.

### Python principles associated with scikit-learn

* Objects
* Classes
* Methods
* Functions
* Variables
* Arrays
* Abstraction

The decoder provides a structured interface for:

```python
decoder.prepare_data(...)
```

```python
decoder.train()
```

```python
decoder.evaluate()
```

```python
decoder.get_predictions()
```

The machine-learning package performs the underlying regression calculations.

---

## pathlib

**Purpose:**

* Managing file paths.
* Identifying the NWB file.
* Working with files and directories.

### Python principles associated with pathlib

* Objects
* Variables
* Methods
* Operators

A path can be represented as a Python object rather than manually constructing a long text path.

The general concept is:

**File location → Path object → File access**

---

## How the Packages Work Together

The packages perform different jobs within the computational pipeline.

**PyNWB**

→ Provides access to the recorded neurophysiology data.

**NumPy**

→ Represents and processes the numerical neural and behavioral measurements.

**scikit-learn**

→ Trains the Ridge decoder and generates predictions.

**pathlib**

→ Handles file locations and paths.

The Python programming principles provide the structure connecting them:

**Variables → Functions → Classes → Objects → Methods → Arrays → Conditions → Data Validation → Data Flow → Machine Learning**

Together, these principles and packages form the computational foundation of the intracortical BCI pipeline.

# 3. Statistics

## R² — Coefficient of Determination

**Definition:** Measures how much of the variation in the actual finger velocity is explained by the decoder's predictions.

**How it was used:** The Ridge decoder calculates R² separately for X and Y finger velocity.

```python
metrics["r2"]
```

**Results:**

```text
X: 0.1285
Y: 0.1062
```

---

## RMSE — Root Mean Squared Error

**Definition:** Measures the typical size of prediction errors while giving larger errors more influence.

**How it was used:** It compares predicted finger velocity against actual recorded finger velocity.

```python
metrics["rmse"]
```

**Results:**

```text
X: 75.4833
Y: 58.7098
```

---

## MAE — Mean Absolute Error

**Definition:** Measures the average absolute difference between predicted and actual finger velocity.

**How it was used:** It provides an error measurement based on the absolute size of each prediction error.

```python
metrics["mae"]
```

**Results:**

```text
X: 43.4004
Y: 37.0207
```

---

## Pearson Correlation

**Definition:** Measures the strength and direction of the linear relationship between predicted and actual finger velocity.

**How it was used:** The pipeline calculates Pearson correlation separately for X and Y velocity.

```python
metrics["correlation_x"]

metrics["correlation_y"]
```

**Results:**

```text
X: 0.3639
Y: 0.3311
```

---

## Training Set

**Definition:** The portion of the neural dataset used to train the Ridge decoder.

**How it was used:** The decoder used 80% of the aligned data for training.

```text
25,952 samples
```

---

## Testing Set

**Definition:** The portion of the dataset kept separate from training and used to evaluate the trained decoder.

**How it was used:** The decoder used 20% of the aligned data for testing.

```text
6,488 samples
```

---

## 80/20 Split

**Definition:** A division where 80% of the data is used for training and 20% is used for testing.

**How it was used:**

```python
decoder = NeuralDecoder(
    test_size=0.20,
    alpha=1.0
)
```

This produced:

```text
Training:
25,952 samples

Testing:
6,488 samples
```

---

## Alpha — Ridge Regularization Parameter

**Definition:** Controls the amount of regularization applied by the Ridge regression model.

**How it was used:**

```python
alpha=1.0
```

The value controls how strongly the model penalizes large regression coefficients.

---

## 20-ms Binning

**Definition:** Groups neural measurements into 20-millisecond time windows.

**How it was used:**

```python
bin_size = 0.020
```

The neural activity is converted into population-level measurements at this temporal resolution before being aligned with finger velocity.

---

## Population Feature Matrix

**Definition:** A matrix containing the activity of all recorded neurons across time.

**How it was used:**

```text
32,455 × 130
```

This means:

```text
32,455 time points
130 neurons
```

Each row represents neural population activity at one point in the recording, while each column represents one neuron.

---

## Behavioral Target Matrix

**Definition:** The numerical movement measurements that the decoder is trained to predict.

**How it was used:**

```text
32,440 × 2
```

The two columns represent:

```text
X finger velocity
Y finger velocity
```

---

## Aligned Dataset

**Definition:** Neural and behavioral measurements that have been matched to the same time bins.

**How it was used:**

```text
32,440 neural samples
32,440 behavioral samples
```

This creates the paired dataset required for supervised machine learning:

```text
X = neural activity

y = finger velocity
```

---

## Pearson X

**Definition:** The Pearson correlation between predicted and actual X-axis finger velocity.

**Result:**

```text
0.3638689504
```

---

## Pearson Y

**Definition:** The Pearson correlation between predicted and actual Y-axis finger velocity.

**Result:**

```text
0.3310680170
```

---

## Summary

The statistics answer different questions:

```text
R²
→ How much variation does the model explain?

RMSE
→ How large are the prediction errors?

MAE
→ What is the average absolute prediction error?

Pearson correlation
→ How strongly do predicted and actual movements vary together?

80/20 split
→ How was the data divided between training and testing?

Alpha
→ How much Ridge regularization was applied?

20-ms bin
→ At what temporal resolution was neural activity represented?

Population feature matrix
→ How was the activity of the neuron population represented?

Behavioral target matrix
→ What movement was the decoder trained to predict?

Aligned dataset
→ How were neural activity and movement paired in time?
```
