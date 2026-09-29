
# ==========================================
# TEMPORAL ALIGNMENT
#
# Align neural population activity with
# continuous behavioral measurements.
# ==========================================

import numpy as np


class TemporalAligner:

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(
        self,
        neural_times,
        neural_features,
        behavior_times,
        behavior_data,
        bin_size=0.020
    ):

        self.neural_times = np.asarray(
            neural_times,
            dtype=float
        )

        self.neural_features = np.asarray(
            neural_features,
            dtype=float
        )

        self.behavior_times = np.asarray(
            behavior_times,
            dtype=float
        )

        self.behavior_data = np.asarray(
            behavior_data,
            dtype=float
        )

        self.bin_size = float(
            bin_size
        )

        self.aligned_neural = None
        self.aligned_behavior = None
        self.aligned_times = None

        self.removed_samples = 0


    # ==========================================
    # VALIDATION
    # ==========================================

    def validate(self):

        if self.bin_size <= 0:

            raise ValueError(
                "bin_size must be greater than zero."
            )


        if self.neural_features.ndim != 2:

            raise ValueError(
                "Neural features must be 2-dimensional."
            )


        if self.behavior_data.ndim != 2:

            raise ValueError(
                "Behavior data must be 2-dimensional."
            )


        if len(self.neural_times) != len(
            self.neural_features
        ):

            raise ValueError(
                "Neural timestamps and neural features "
                "have different lengths."
            )


        if len(self.behavior_times) != len(
            self.behavior_data
        ):

            raise ValueError(
                "Behavior timestamps and behavior data "
                "have different lengths."
            )


        if len(self.neural_times) == 0:

            raise ValueError(
                "No neural data was provided."
            )


        if len(self.behavior_times) == 0:

            raise ValueError(
                "No behavioral data was provided."
            )


        if np.any(
            np.diff(self.neural_times) < 0
        ):

            raise ValueError(
                "Neural timestamps must be "
                "monotonically increasing."
            )


        if np.any(
            np.diff(self.behavior_times) < 0
        ):

            raise ValueError(
                "Behavior timestamps must be "
                "monotonically increasing."
            )


        if self.neural_features.shape[1] == 0:

            raise ValueError(
                "Neural feature matrix contains "
                "zero neurons."
            )


        if self.behavior_data.shape[1] == 0:

            raise ValueError(
                "Behavior data contains zero dimensions."
            )


    # ==========================================
    # REMOVE NON-FINITE SAMPLES
    # ==========================================

    def _remove_invalid_samples(self):

        if (
            self.aligned_neural is None
            or self.aligned_behavior is None
            or self.aligned_times is None
        ):

            raise ValueError(
                "Alignment must be performed before "
                "removing invalid samples."
            )


        neural_valid = np.all(
            np.isfinite(
                self.aligned_neural
            ),
            axis=1
        )


        behavior_valid = np.all(
            np.isfinite(
                self.aligned_behavior
            ),
            axis=1
        )


        time_valid = np.isfinite(
            self.aligned_times
        )


        valid_mask = (
            neural_valid
            &
            behavior_valid
            &
            time_valid
        )


        self.removed_samples = int(
            np.sum(
                ~valid_mask
            )
        )


        self.aligned_neural = (
            self.aligned_neural[
                valid_mask
            ]
        )


        self.aligned_behavior = (
            self.aligned_behavior[
                valid_mask
            ]
        )


        self.aligned_times = (
            self.aligned_times[
                valid_mask
            ]
        )


    # ==========================================
    # ALIGN BY BIN
    # ==========================================

    def align_by_bin(self):

        self.validate()


        half_bin = (
            self.bin_size / 2.0
        )


        bin_starts = (
            self.neural_times
            -
            half_bin
        )


        bin_ends = (
            self.neural_times
            +
            half_bin
        )


        number_of_bins = (
            len(self.neural_times)
        )


        behavior_dimensions = (
            self.behavior_data.shape[1]
        )


        aligned_behavior = np.full(
            (
                number_of_bins,
                behavior_dimensions
            ),
            np.nan,
            dtype=float
        )


        for bin_index in range(
            number_of_bins
        ):

            start = (
                bin_starts[bin_index]
            )

            end = (
                bin_ends[bin_index]
            )


            mask = (
                (self.behavior_times >= start)
                &
                (self.behavior_times < end)
            )


            if not np.any(mask):

                continue


            values = (
                self.behavior_data[mask]
            )


            aligned_behavior[
                bin_index
            ] = np.mean(
                values,
                axis=0
            )


        valid_mask = ~np.any(
            np.isnan(
                aligned_behavior
            ),
            axis=1
        )


        self.aligned_times = (
            self.neural_times[
                valid_mask
            ]
        )


        self.aligned_neural = (
            self.neural_features[
                valid_mask
            ]
        )


        self.aligned_behavior = (
            aligned_behavior[
                valid_mask
            ]
        )


        self._remove_invalid_samples()


        return (
            self.aligned_times,
            self.aligned_neural,
            self.aligned_behavior
        )


    # ==========================================
    # ALIGN BY NEAREST SAMPLE
    # ==========================================

    def align_nearest(self):

        self.validate()


        right_indices = np.searchsorted(
            self.behavior_times,
            self.neural_times,
            side="left"
        )


        right_indices = np.clip(
            right_indices,
            0,
            len(self.behavior_times) - 1
        )


        left_indices = np.maximum(
            right_indices - 1,
            0
        )


        left_distance = np.abs(
            self.behavior_times[
                left_indices
            ]
            -
            self.neural_times
        )


        right_distance = np.abs(
            self.behavior_times[
                right_indices
            ]
            -
            self.neural_times
        )


        nearest_indices = np.where(
            left_distance < right_distance,
            left_indices,
            right_indices
        )


        self.aligned_times = (
            self.neural_times.copy()
        )


        self.aligned_neural = (
            self.neural_features.copy()
        )


        self.aligned_behavior = (
            self.behavior_data[
                nearest_indices
            ].copy()
        )


        # --------------------------------------
        # REMOVE NON-FINITE VALUES
        # --------------------------------------

        self._remove_invalid_samples()


        return (
            self.aligned_times,
            self.aligned_neural,
            self.aligned_behavior
        )


    # ==========================================
    # GET NEURAL DATA
    # ==========================================

    def get_neural_data(self):

        if self.aligned_neural is None:

            raise ValueError(
                "Alignment has not been performed."
            )

        return self.aligned_neural


    # ==========================================
    # GET BEHAVIOR DATA
    # ==========================================

    def get_behavior_data(self):

        if self.aligned_behavior is None:

            raise ValueError(
                "Alignment has not been performed."
            )

        return self.aligned_behavior


    # ==========================================
    # GET TIMES
    # ==========================================

    def get_times(self):

        if self.aligned_times is None:

            raise ValueError(
                "Alignment has not been performed."
            )

        return self.aligned_times


    # ==========================================
    # GET SHAPES
    # ==========================================

    def get_shapes(self):

        if (
            self.aligned_neural is None
            or self.aligned_behavior is None
            or self.aligned_times is None
        ):

            raise ValueError(
                "Alignment has not been performed."
            )


        return {

            "neural_shape":
                self.aligned_neural.shape,

            "behavior_shape":
                self.aligned_behavior.shape,

            "time_shape":
                self.aligned_times.shape

        }


    # ==========================================
    # GET REMOVED SAMPLE COUNT
    # ==========================================

    def get_removed_sample_count(self):

        return self.removed_samples
