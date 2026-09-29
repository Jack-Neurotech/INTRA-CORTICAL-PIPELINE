
# ==========================================
# INTRACORTICAL BCI DECODER
#
# MAIN PIPELINE
#
# NWB
#   ↓
# Neural Data
#   ↓
# Behavior
#   ↓
# Population Features
#   ↓
# Temporal Alignment
#   ↓
# Ridge Neural Decoder
#   ↓
# Evaluation
#   ↓
# Scientific HTML Report
# ==========================================

from nwb_reader import NWBReader
from neural_data import NeuralData
from behavior import BehaviorData
from features import FeatureExtractor
from alignment import TemporalAligner
from decoder import NeuralDecoder
from report import generate_report


# ==========================================
# LOAD NWB FILE
# ==========================================

print()
print("==========================================")
print("INTRACORTICAL BCI DECODER")
print("==========================================")

reader = NWBReader()

nwbfile = reader.load()


# ==========================================
# NWB FILE
# ==========================================

print()
print("==========================================")
print("NWB FILE")
print("==========================================")

print()
print("Loaded:")
print(reader.file_path)


# ==========================================
# NEURAL DATA
# ==========================================

neural_data = NeuralData(
    nwbfile
)

neural_data.extract_units()

number_of_neurons = (
    neural_data.get_number_of_units()
)

recording_duration = (
    neural_data.get_recording_duration()
)


print()
print("==========================================")
print("NEURAL DATA")
print("==========================================")

print()
print("Number of neurons:")
print(number_of_neurons)

print()
print("Recording duration:")
print(
    recording_duration,
    "seconds"
)


# ==========================================
# BEHAVIOR DATA
# ==========================================

behavior_data = BehaviorData(
    nwbfile
)

behavior_data.extract_trials()

number_of_trials = (
    behavior_data.get_number_of_trials()
)


print()
print("==========================================")
print("BEHAVIOR DATA")
print("==========================================")

print()
print("Number of trials:")
print(number_of_trials)

print()
print("Trial columns:")
print(
    behavior_data.get_trial_columns()
)


# ==========================================
# DISCOVER CONTINUOUS BEHAVIOR
# ==========================================

behavior_objects = (
    behavior_data.discover_behavior_objects()
)


print()
print("==========================================")
print("CONTINUOUS BEHAVIOR")
print("==========================================")

for name in behavior_objects:
    print(name)


# ==========================================
# FEATURE EXTRACTION
# ==========================================

feature_extractor = FeatureExtractor(
    neural_data=neural_data,
    behavior_data=behavior_data
)


# ==========================================
# TRIAL FEATURES
# ==========================================

trial_features = (
    feature_extractor.create_trial_features()
)


# ==========================================
# POPULATION FEATURES
# ==========================================

(
    population_times,
    population_features
) = (
    feature_extractor.create_population_features(
        bin_size=0.020
    )
)


# ==========================================
# BEHAVIOR TARGET
# ==========================================

(
    behavior_times,
    finger_velocity
) = (
    feature_extractor.create_behavior_target(
        "finger_vel"
    )
)


print()
print("==========================================")
print("FEATURE EXTRACTION")
print("==========================================")

print()
print("Trial feature matrix:")
print(
    trial_features.shape
)

print()
print("Population feature matrix:")
print(
    population_features.shape
)

print()
print("Population time range:")
print(
    population_times[0],
    "→",
    population_times[-1],
    "seconds"
)

print()
print("Behavior target:")
print("finger_vel")

print()
print("Behavior target shape:")
print(
    finger_velocity.shape
)

print()
print("Behavior time range:")
print(
    behavior_times[0],
    "→",
    behavior_times[-1],
    "seconds"
)


# ==========================================
# TEMPORAL ALIGNMENT
# ==========================================

aligner = TemporalAligner(
    neural_times=population_times,
    neural_features=population_features,
    behavior_times=behavior_times,
    behavior_data=finger_velocity,
    bin_size=0.020
)


(
    aligned_times,
    aligned_neural,
    aligned_behavior
) = aligner.align_nearest()


print()
print("==========================================")
print("TEMPORAL ALIGNMENT")
print("==========================================")

print()
print("Aligned neural data:")
print(
    aligned_neural.shape
)

print()
print("Aligned behavioral data:")
print(
    aligned_behavior.shape
)

print()
print("Aligned time data:")
print(
    aligned_times.shape
)

print()
print("Aligned time range:")
print(
    aligned_times[0],
    "→",
    aligned_times[-1],
    "seconds"
)

print()
print("Final aligned dataset:")
print(
    aligned_neural.shape
)

print()
print("Final behavioral target:")
print(
    aligned_behavior.shape
)


# ==========================================
# NEURAL DECODER
# ==========================================

decoder = NeuralDecoder(
    test_size=0.20,
    alpha=1.0
)


# ==========================================
# PREPARE DATA
# ==========================================

(
    X_train,
    X_test,
    y_train,
    y_test
) = decoder.prepare_data(
    aligned_neural,
    aligned_behavior
)


print()
print("==========================================")
print("NEURAL DECODER")
print("==========================================")

print()
print("Training features:")
print(
    X_train.shape
)

print()
print("Testing features:")
print(
    X_test.shape
)

print()
print("Training targets:")
print(
    y_train.shape
)

print()
print("Testing targets:")
print(
    y_test.shape
)


# ==========================================
# TRAIN
# ==========================================

print()
print("Training Ridge decoder...")

decoder.train()

print("Training complete.")


# ==========================================
# PREDICT
# ==========================================

predictions = decoder.predict(
    X_test
)


# ==========================================
# EVALUATION
# ==========================================

metrics = decoder.evaluate()


print()
print("==========================================")
print("DECODER EVALUATION")
print("==========================================")

print()
print("R²:")
print(
    metrics["r2"]
)

print()
print("RMSE:")
print(
    metrics["rmse"]
)

print()
print("MAE:")
print(
    metrics["mae"]
)

print()
print("Pearson correlation X:")
print(
    metrics["correlation_x"]
)

print()
print("Pearson correlation Y:")
print(
    metrics["correlation_y"]
)

print()
print("Predictions:")
print(
    predictions.shape
)


# ==========================================
# PIPELINE SUMMARY
# ==========================================

print()
print("==========================================")
print("PIPELINE SUMMARY")
print("==========================================")

print()
print("Neurons:")
print(number_of_neurons)

print()
print("Trials:")
print(number_of_trials)

print()
print("Recording duration:")
print(recording_duration)

print()
print("Population features:")
print(
    population_features.shape
)

print()
print("Aligned neural data:")
print(
    aligned_neural.shape
)

print()
print("Aligned behavior:")
print(
    aligned_behavior.shape
)

print()
print("Training set:")
print(
    X_train.shape
)

print()
print("Testing set:")
print(
    X_test.shape
)

print()
print("Predictions:")
print(
    predictions.shape
)


# ==========================================
# FINAL DECODER RESULTS
# ==========================================

print()
print("==========================================")
print("FINAL DECODER RESULTS")
print("==========================================")


# ==========================================
# X VELOCITY
# ==========================================

print()
print("Finger velocity X")
print("-----------------")

print()
print("R²:")
print(
    metrics["r2"][0]
)

print()
print("RMSE:")
print(
    metrics["rmse"][0]
)

print()
print("MAE:")
print(
    metrics["mae"][0]
)

print()
print("Pearson:")
print(
    metrics["correlation_x"]
)


# ==========================================
# Y VELOCITY
# ==========================================

print()
print("Finger velocity Y")
print("-----------------")

print()
print("R²:")
print(
    metrics["r2"][1]
)

print()
print("RMSE:")
print(
    metrics["rmse"][1]
)

print()
print("MAE:")
print(
    metrics["mae"][1]
)

print()
print("Pearson:")
print(
    metrics["correlation_y"]
)


# ==========================================
# SCIENTIFIC REPORT
# ==========================================

print()
print("==========================================")
print("GENERATING SCIENTIFIC REPORT")
print("==========================================")


report_path = generate_report(
    output_directory="reports",
    dataset_path=reader.file_path,

    number_of_neurons=number_of_neurons,
    number_of_trials=number_of_trials,
    recording_duration=recording_duration,

    population_times=population_times,
    population_features=population_features,

    aligned_times=aligned_times,
    aligned_neural=aligned_neural,
    aligned_behavior=aligned_behavior,

    y_test=y_test,
    predictions=predictions,

    metrics=metrics,

    training_shape=X_train.shape,
    testing_shape=X_test.shape
)


print()
print("Report created:")
print(
    report_path
)

print()
print("The scientific report has been opened")
print("in your default browser.")


# ==========================================
# CLOSE NWB FILE
# ==========================================

reader.close()


# ==========================================
# COMPLETE
# ==========================================

print()
print("==========================================")
print("DECODER PIPELINE COMPLETE")
print("==========================================")
