1. What does the Intracortical BCI Pipeline do?
Reads recorded intracortical neural data from an NWB neurophysiology file.
Extracts neural activity from the recorded population of neurons.
Extracts behavioral data recorded at the same time, including finger velocity.
Converts neural activity into numerical features that can be used by a machine-learning model.
Bins the neural activity into 20-ms time windows so that neural activity can be represented consistently over time.
Temporally aligns the neural activity with the behavioral measurements, making sure that each neural observation corresponds to the appropriate behavioral observation.
Uses the activity of the neuron population as the input to a decoder.
Uses finger velocity as the target that the decoder is trying to predict.
Trains a Ridge regression model to learn the relationship between neural population activity and finger movement.
Predicts finger velocity from neural activity, specifically:
X-axis finger velocity
Y-axis finger velocity
Evaluates the predictions by comparing the predicted movement against the actual recorded movement.
Calculates quantitative performance measures, including:
R²
RMSE
MAE
Pearson correlation
In simple terms, the project takes:
recorded activity from neurons → processes the neural activity → connects it to recorded movement → trains a decoder → predicts movement from neural activity.

2. How it was Coded:

   
# PYTHON PRINCIPLES

## Variables

**Definition:** A variable stores information under a name.

**How it was used:** Variables store the NWB recording, neural data, behavioral data, feature matrices, decoder settings, and results.

```python
bin_size = 0.020

behavior_target_name = "finger_vel"

number_of_neurons = neural_data.get_number_of_units()
```

---

## Functions

**Definition:** A function is a reusable block of code that performs a specific task.

**How it was used:** Functions divide the pipeline into individual operations such as loading data, extracting neurons, creating features, aligning data, training the decoder, and evaluating results.

```python
nwbfile = reader.load()

neural_data.extract_units()

behavior_data.extract_trials()

aligned_times, aligned_neural, aligned_behavior = (
    aligner.align_by_bin()
)

decoder.train()

metrics = decoder.evaluate()
```

---

## Classes

**Definition:** A class is a blueprint for creating objects that contain related data and functions.

**How it was used:** The project separates the different stages of the pipeline into specialized classes.

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

Each class has a specific responsibility rather than putting the entire pipeline into one large block of code.

---

## Objects

**Definition:** An object is a specific instance created from a class.

**How it was used:** Objects allow each part of the pipeline to maintain its own data and perform operations on that data.

```python
decoder = NeuralDecoder(
    test_size=0.20,
    alpha=1.0
)
```

The `decoder` object contains the settings and methods needed to prepare, train, and evaluate the neural decoder.

---

## Methods

**Definition:** A method is a function that belongs to an object.

**How it was used:** The pipeline uses methods to perform operations on specific components.

```python
neural_data.extract_units()
```

```python
behavior_data.extract_trials()
```

```python
feature_extractor.create_population_features(
    bin_size=0.020
)
```

```python
decoder.train()
```

The method operates on the object it belongs to.

---

## Arrays

**Definition:** An array is an organized collection of numerical values.

**How it was used:** Neural recordings and behavioral recordings contain large amounts of numerical data, so the pipeline represents them as NumPy arrays.

For example:

```python
population_features
```

contains the neural population activity.

The resulting matrix was:

```text
32,455 × 130
```

meaning:

```text
32,455 time points
130 neurons
```

The behavioral target was represented as:

```text
32,440 × 2
```

meaning:

```text
32,440 time points
2 movement dimensions
```

---

## Indexing

**Definition:** Indexing accesses a specific element or section of a data structure.

**How it was used:** The pipeline accesses individual time points, neurons, trials, and prediction values through indexing.

```python
aligned_times[0]
```

gets the first aligned time point.

```python
aligned_times[-1]
```

gets the final aligned time point.

---

## Conditional Statements

**Definition:** Conditional statements allow Python to make decisions based on a condition.

**How it was used:** The pipeline checks whether the data are valid before sending them into the decoder.

```python
if not np.all(valid_mask):

    aligned_times = aligned_times[
        valid_mask
    ]

    aligned_neural = aligned_neural[
        valid_mask
    ]

    aligned_behavior = aligned_behavior[
        valid_mask
    ]
```

This ensures that invalid measurements are removed before machine learning.

---

## Boolean Logic

**Definition:** Boolean logic evaluates conditions as `True` or `False`.

**How it was used:** The pipeline combines multiple conditions when determining whether neural and behavioral measurements are valid.

```python
np.all(
    np.isfinite(aligned_neural),
    axis=1
)
```

checks whether all neural features in a row contain valid finite numbers.

The pipeline also checks the behavioral data:

```python
np.all(
    np.isfinite(aligned_behavior),
    axis=1
)
```

The two conditions are combined with:

```python
&
```

so that a time point is retained only when both neural and behavioral data are valid.

---

## Loops

**Definition:** A loop repeats code for multiple pieces of information.

**How it was used:** Loops allow the pipeline to process collections of trials, behavioral variables, neurons, and other recorded information without manually writing the same operation repeatedly.

For example:

```python
for name in behavior_objects:

    print(name)
```

prints each discovered behavioral object.

---

## Data Validation

**Definition:** Data validation checks whether information meets the requirements of the program before it is processed.

**How it was used:** The pipeline checks the neural and behavioral data before training the decoder.

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

This prevents `NaN` and infinite values from being passed into the model.

---

## Object-Oriented Programming

**Definition:** Object-oriented programming organizes a program into objects that contain related data and operations.

**How it was used:** The entire pipeline is divided into specialized objects.

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

## Modular Programming

**Definition:** Modular programming divides a large program into smaller independent components.

**How it was used:** Instead of putting the entire intracortical decoder into one Python file, the project separates functionality into modules such as:

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

allows the main pipeline to use the neural-data module without rewriting its implementation.

---

## Data Flow

**Definition:** Data flow is the movement and transformation of information through a program.

**How it was used:** The project passes data from one processing stage to the next.

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
Decoder
   ↓
Predictions
   ↓
Evaluation metrics
```

The Python code therefore acts as a series of transformations, where the output from one stage becomes the input to the next.

---

## Exception Handling

**Definition:** Exceptions allow a program to identify conditions where it cannot safely continue.

**How it was used:** The project uses errors to prevent invalid or missing data from silently entering the pipeline.

For example, the behavioral-data component checks whether a trial table exists before attempting to use it.

```python
if self.nwbfile.trials is None:

    raise ValueError(
        "The NWB file does not contain "
        "a trial table."
    )
```

This makes failures explicit instead of allowing incorrect data to propagate through the analysis.

---

## Numerical Computing

**Definition:** Numerical computing means using mathematical operations on numerical data.

**How it was used:** This is fundamental to the intracortical pipeline because neural recordings are numerical measurements.

NumPy is used to perform operations such as:

```python
np.asarray(...)
```

```python
np.isfinite(...)
```

and array-based calculations.

The numerical data ultimately become the input to the machine-learning model.

---

## Machine-Learning Abstraction

**Definition:** Abstraction means using a higher-level object or function without needing to manually implement every underlying mathematical operation.

**How it was used:** Instead of manually implementing Ridge regression mathematics, the decoder uses a machine-learning model through the project's decoder class.

```python
decoder = NeuralDecoder(
    test_size=0.20,
    alpha=1.0
)
```

The decoder then handles:

```python
decoder.prepare_data(...)

decoder.train()

decoder.evaluate()

decoder.get_predictions()
```

This allows the project to focus on the **neuroscience data pipeline and interpretation** while the machine-learning library handles the underlying regression implementation. 

3. Explain Statistics:

   # STATISTICS

## R² — Coefficient of Determination

**Definition:** Measures how much of the variation in the actual finger velocity is explained by the decoder's predictions.

**How it was used:** The Ridge decoder calculates R² separately for X and Y finger velocity.

```python
metrics["r2"]
```

Your results:

```text
X: 0.1285
Y: 0.1062
```

---

## RMSE — Root Mean Squared Error

**Definition:** Measures the typical size of the prediction error while giving larger errors more influence.

**How it was used:** It compares the predicted finger velocity against the actual finger velocity.

```python
metrics["rmse"]
```

Your results:

```text
X: 75.4833
Y: 58.7098
```

---

## MAE — Mean Absolute Error

**Definition:** Measures the average absolute difference between the predicted and actual finger velocity.

**How it was used:** It provides an error measurement that is easier to interpret than RMSE because every error contributes according to its absolute size.

```python
metrics["mae"]
```

Your results:

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

Your results:

```text
X: 0.3639
Y: 0.3311
```

---

## Training Set

**Definition:** The portion of the neural dataset used to teach the Ridge decoder the relationship between neural activity and finger velocity.

**How it was used:** Your decoder used 80% of the aligned data for training.

```text
25,952 samples
```

---

## Testing Set

**Definition:** The portion of the dataset that is kept separate from training and used to evaluate the trained decoder.

**How it was used:** Your decoder used 20% of the aligned data for testing.

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

The neural activity is converted into population-level measurements at this time resolution before being aligned with finger velocity.

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
0.3310680174
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
```

