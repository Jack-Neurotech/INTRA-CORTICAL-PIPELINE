
# ==========================================
# NEURAL FEATURE EXTRACTION
#
# Trial-Level Features
# +
# Time-Binned Population Features
# ==========================================

import numpy as np


class FeatureExtractor:

    def __init__(
        self,
        neural_data,
        behavior_data
    ):

        self.neural_data = neural_data
        self.behavior_data = behavior_data

        # --------------------------------------
        # EXISTING TRIAL-LEVEL FEATURES
        # --------------------------------------

        self.features = None
        self.labels = None
        self.trial_indices = None

        # --------------------------------------
        # TIME-BINNED POPULATION FEATURES
        # --------------------------------------

        self.population_features = None
        self.population_times = None

        # --------------------------------------
        # BEHAVIOR TARGET
        # --------------------------------------

        self.behavior_target = None
        self.behavior_target_times = None


    # ==========================================
    # EXISTING TRIAL FEATURE EXTRACTION
    # ==========================================

    def create_trial_features(
        self,
        start_time_column="start_time",
        stop_time_column="stop_time"
    ):

        start_times = (
            self.behavior_data.get_start_times()
        )

        stop_times = (
            self.behavior_data.get_stop_times()
        )

        unit_ids = (
            self.neural_data.get_unit_ids()
        )

        number_of_trials = len(
            start_times
        )

        number_of_units = len(
            unit_ids
        )

        features = np.zeros(
            (
                number_of_trials,
                number_of_units
            ),
            dtype=float
        )


        # --------------------------------------
        # COUNT SPIKES PER TRIAL
        # --------------------------------------

        for trial_index in range(
            number_of_trials
        ):

            trial_start = float(
                start_times[trial_index]
            )

            trial_stop = float(
                stop_times[trial_index]
            )


            for neuron_index, unit_id in enumerate(
                unit_ids
            ):

                spike_times = (
                    self.neural_data.get_spike_times(
                        unit_id
                    )
                )

                spikes_in_trial = (
                    spike_times[
                        (spike_times >= trial_start)
                        &
                        (spike_times < trial_stop)
                    ]
                )

                features[
                    trial_index,
                    neuron_index
                ] = len(
                    spikes_in_trial
                )


        self.features = features

        self.trial_indices = np.arange(
            number_of_trials
        )

        return self.features


    # ==========================================
    # GET BEHAVIORAL LABELS
    # ==========================================

    def get_labels(
        self,
        variable_name
    ):

        labels = (
            self.behavior_data.get_variable(
                variable_name
            )
        )

        if self.features is not None:

            if len(labels) != len(
                self.features
            ):

                raise ValueError(
                    "Number of behavioral labels "
                    "does not match number of "
                    "trial feature vectors."
                )


        self.labels = np.asarray(
            labels
        )

        return self.labels


    # ==========================================
    # GET EXISTING FEATURES
    # ==========================================

    def get_features(self):

        if self.features is None:

            raise ValueError(
                "Trial features have not "
                "been created yet."
            )

        return self.features


    # ==========================================
    # GET LABEL ARRAY
    # ==========================================

    def get_labels_array(self):

        if self.labels is None:

            raise ValueError(
                "Behavioral labels have not "
                "been extracted yet."
            )

        return self.labels


    # ==========================================
    # CONVERT COUNTS TO FIRING RATES
    # ==========================================

    def counts_to_firing_rates(self):

        if self.features is None:

            raise ValueError(
                "Trial features have not "
                "been created yet."
            )

        start_times = (
            self.behavior_data.get_start_times()
        )

        stop_times = (
            self.behavior_data.get_stop_times()
        )

        durations = (
            stop_times - start_times
        )


        if np.any(
            durations <= 0
        ):

            raise ValueError(
                "One or more trials have "
                "zero or negative duration."
            )


        firing_rates = (
            self.features
            /
            durations[:, np.newaxis]
        )

        return firing_rates


    # ==========================================
    # TIME-BINNED POPULATION ACTIVITY
    # ==========================================
    #
    # Converts individual neuron spike trains
    # into a population activity matrix.
    #
    # Example:
    #
    # 20 ms bins
    #
    #        N1 N2 N3 ... N130
    #
    # 0 ms    1  0  2  ...  0
    # 20 ms   0  1  1  ...  2
    # 40 ms   2  0  0  ...  1
    #
    # ==========================================

    def create_population_features(
        self,
        bin_size=0.020,
        start_time=None,
        stop_time=None
    ):

        bin_size = float(
            bin_size
        )


        if bin_size <= 0:

            raise ValueError(
                "bin_size must be greater than zero."
            )


        unit_ids = (
            self.neural_data.get_unit_ids()
        )


        # --------------------------------------
        # DETERMINE RECORDING WINDOW
        # --------------------------------------

        if start_time is None:

            start_time = 0.0

        else:

            start_time = float(
                start_time
            )


        if stop_time is None:

            stop_time = (
                self.neural_data
                .get_recording_duration()
            )

        else:

            stop_time = float(
                stop_time
            )


        if stop_time <= start_time:

            raise ValueError(
                "stop_time must be greater "
                "than start_time."
            )


        # --------------------------------------
        # CREATE TIME BINS
        # --------------------------------------

        number_of_bins = int(
            np.ceil(
                (stop_time - start_time)
                /
                bin_size
            )
        )


        bin_edges = (
            start_time
            +
            np.arange(
                number_of_bins + 1
            )
            *
            bin_size
        )


        bin_times = (
            bin_edges[:-1]
            +
            bin_size / 2.0
        )


        # --------------------------------------
        # CREATE POPULATION MATRIX
        # --------------------------------------

        population_features = np.zeros(
            (
                number_of_bins,
                len(unit_ids)
            ),
            dtype=float
        )


        # --------------------------------------
        # BIN EACH NEURON
        # --------------------------------------

        for neuron_index, unit_id in enumerate(
            unit_ids
        ):

            spike_times = (
                self.neural_data.get_spike_times(
                    unit_id
                )
            )


            spike_times = np.asarray(
                spike_times,
                dtype=float
            )


            # Only use spikes inside
            # requested recording window.

            spike_times = spike_times[
                (spike_times >= start_time)
                &
                (spike_times < stop_time)
            ]


            if len(spike_times) == 0:

                continue


            spike_counts, _ = np.histogram(
                spike_times,
                bins=bin_edges
            )


            population_features[
                :,
                neuron_index
            ] = spike_counts


        self.population_features = (
            population_features
        )

        self.population_times = (
            bin_times
        )


        return (
            self.population_times,
            self.population_features
        )


    # ==========================================
    # GET POPULATION FEATURES
    # ==========================================

    def get_population_features(self):

        if self.population_features is None:

            raise ValueError(
                "Population features have "
                "not been created yet."
            )

        return self.population_features


    # ==========================================
    # GET POPULATION TIMES
    # ==========================================

    def get_population_times(self):

        if self.population_times is None:

            raise ValueError(
                "Population feature times "
                "have not been created yet."
            )

        return self.population_times


    # ==========================================
    # EXTRACT CONTINUOUS BEHAVIOR TARGET
    # ==========================================

    def create_behavior_target(
        self,
        signal_name
    ):

        signal = (
            self.behavior_data.get_signal(
                signal_name
            )
        )


        self.behavior_target = (
            np.asarray(
                signal["data"]
            )
        )

        self.behavior_target_times = (
            np.asarray(
                signal["timestamps"],
                dtype=float
            )
        )


        if (
            len(self.behavior_target)
            !=
            len(self.behavior_target_times)
        ):

            raise ValueError(
                "Behavior target data and "
                "timestamps have different lengths."
            )


        return (
            self.behavior_target_times,
            self.behavior_target
        )


    # ==========================================
    # GET BEHAVIOR TARGET
    # ==========================================

    def get_behavior_target(self):

        if self.behavior_target is None:

            raise ValueError(
                "Behavior target has "
                "not been created yet."
            )

        return self.behavior_target


    # ==========================================
    # GET BEHAVIOR TARGET TIMES
    # ==========================================

    def get_behavior_target_times(self):

        if (
            self.behavior_target_times
            is None
        ):

            raise ValueError(
                "Behavior target timestamps "
                "have not been created yet."
            )

        return self.behavior_target_times

