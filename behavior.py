
# ==========================================
# INTRACORTICAL BCI DECODER
#
# BEHAVIORAL DATA EXTRACTION
#
# PURPOSE:
#
# Extract:
#
#     Trial metadata
#     Continuous behavioral signals
#
# Examples:
#
#     cursor position
#     finger position
#     velocity
#     target position
#
# ==========================================

import numpy as np


class BehaviorData:

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(self, nwbfile):

        self.nwbfile = nwbfile

        self.trials = None
        self.trial_columns = []
        self.number_of_trials = 0

        self.behavior_objects = {}


    # ==========================================
    # TRIAL TABLE
    # ==========================================

    def extract_trials(self):

        if self.nwbfile.trials is None:

            raise ValueError(
                "The NWB file does not contain "
                "a trial table."
            )

        self.trials = self.nwbfile.trials

        self.trial_columns = list(
            self.trials.colnames
        )

        self.number_of_trials = len(
            self.trials.id[:]
        )

        return self.trials


    # ==========================================
    # TRIAL COLUMNS
    # ==========================================

    def get_trial_columns(self):

        if self.trials is None:
            self.extract_trials()

        return self.trial_columns


    # ==========================================
    # NUMBER OF TRIALS
    # ==========================================

    def get_number_of_trials(self):

        if self.trials is None:
            self.extract_trials()

        return self.number_of_trials


    # ==========================================
    # START TIMES
    # ==========================================

    def get_start_times(self):

        if self.trials is None:
            self.extract_trials()

        return np.asarray(
            self.trials["start_time"][:],
            dtype=float
        )


    # ==========================================
    # STOP TIMES
    # ==========================================

    def get_stop_times(self):

        if self.trials is None:
            self.extract_trials()

        return np.asarray(
            self.trials["stop_time"][:],
            dtype=float
        )


    # ==========================================
    # TRIAL VARIABLE
    # ==========================================

    def get_variable(self, variable_name):

        if self.trials is None:
            self.extract_trials()

        if variable_name not in self.trial_columns:

            raise KeyError(
                f"Behavioral variable "
                f"'{variable_name}' was not found "
                f"in the trial table."
            )

        return np.asarray(
            self.trials[variable_name][:]
        )


    # ==========================================
    # GET ONE TRIAL
    # ==========================================

    def get_trial(self, trial_index):

        if self.trials is None:
            self.extract_trials()

        if (
            trial_index < 0
            or
            trial_index >= self.number_of_trials
        ):

            raise IndexError(
                "Trial index is outside the "
                "available trial range."
            )

        trial = {}

        for column in self.trial_columns:

            trial[column] = (
                self.trials[column][trial_index]
            )

        return trial


    # ==========================================
    # GET ALL TRIALS
    # ==========================================

    def get_all_trials(self):

        if self.trials is None:
            self.extract_trials()

        trial_data = {}

        for column in self.trial_columns:

            trial_data[column] = np.asarray(
                self.trials[column][:]
            )

        return trial_data


    # ==========================================
    # DISCOVER PROCESSING MODULES
    # ==========================================

    def discover_processing_modules(self):

        return {
            name: module
            for name, module
            in self.nwbfile.processing.items()
        }


    # ==========================================
    # RECURSIVE NWB OBJECT SEARCH
    # ==========================================

    def _search_container(
        self,
        container,
        path
    ):

        # --------------------------------------
        # Record this object
        # --------------------------------------

        if path:

            self.behavior_objects[path] = container


        # --------------------------------------
        # Search children
        # --------------------------------------

        if hasattr(
            container,
            "children"
        ):

            for child in container.children:

                child_name = getattr(
                    child,
                    "name",
                    "unnamed"
                )

                child_path = (
                    f"{path}/{child_name}"
                    if path
                    else child_name
                )

                self._search_container(
                    child,
                    child_path
                )


        # --------------------------------------
        # Search container mappings
        # --------------------------------------

        if hasattr(
            container,
            "data_interfaces"
        ):

            for name, child in (
                container.data_interfaces.items()
            ):

                child_path = (
                    f"{path}/{name}"
                    if path
                    else name
                )

                self._search_container(
                    child,
                    child_path
                )


        # --------------------------------------
        # Search spatial series
        # --------------------------------------

        if hasattr(
            container,
            "spatial_series"
        ):

            series = container.spatial_series

            if series is not None:

                for name, child in series.items():

                    child_path = (
                        f"{path}/{name}"
                        if path
                        else name
                    )

                    self._search_container(
                        child,
                        child_path
                    )


        # --------------------------------------
        # Search time series
        # --------------------------------------

        if hasattr(
            container,
            "time_series"
        ):

            series = container.time_series

            if series is not None:

                for name, child in series.items():

                    child_path = (
                        f"{path}/{name}"
                        if path
                        else name
                    )

                    self._search_container(
                        child,
                        child_path
                    )


    # ==========================================
    # DISCOVER BEHAVIOR OBJECTS
    # ==========================================

    def discover_behavior_objects(self):

        self.behavior_objects = {}

        modules = (
            self.discover_processing_modules()
        )

        for module_name, module in modules.items():

            self._search_container(
                module,
                module_name
            )

        return self.behavior_objects


    # ==========================================
    # GET BEHAVIORAL OBJECT NAMES
    # ==========================================

    def get_behavior_object_names(self):

        if not self.behavior_objects:

            self.discover_behavior_objects()

        return list(
            self.behavior_objects.keys()
        )


    # ==========================================
    # FIND OBJECT BY NAME
    # ==========================================

    def find_behavior_signal(
        self,
        signal_name
    ):

        if not self.behavior_objects:

            self.discover_behavior_objects()

        signal_name = signal_name.lower()

        for path, obj in (
            self.behavior_objects.items()
        ):

            object_name = getattr(
                obj,
                "name",
                ""
            )

            if (
                object_name
                and
                object_name.lower()
                == signal_name
            ):

                return obj

            final_name = path.split("/")[-1]

            if (
                final_name.lower()
                == signal_name
            ):

                return obj


        raise KeyError(
            f"Behavioral signal "
            f"'{signal_name}' was not found."
        )


    # ==========================================
    # GET SIGNAL DATA
    # ==========================================

    def get_signal_data(
        self,
        signal_name
    ):

        signal = self.find_behavior_signal(
            signal_name
        )

        if not hasattr(
            signal,
            "data"
        ):

            raise TypeError(
                f"'{signal_name}' does not "
                f"contain a data array."
            )

        return np.asarray(
            signal.data[:]
        )


    # ==========================================
    # GET SIGNAL TIMESTAMPS
    # ==========================================

    def get_signal_timestamps(
        self,
        signal_name
    ):

        signal = self.find_behavior_signal(
            signal_name
        )


        # --------------------------------------
        # Explicit timestamps
        # --------------------------------------

        if (
            hasattr(signal, "timestamps")
            and
            signal.timestamps is not None
        ):

            return np.asarray(
                signal.timestamps[:],
                dtype=float
            )


        # --------------------------------------
        # Starting time + sampling rate
        # --------------------------------------

        if (
            hasattr(signal, "starting_time")
            and
            signal.starting_time is not None
            and
            hasattr(signal, "rate")
            and
            signal.rate is not None
        ):

            number_of_samples = len(
                signal.data
            )

            return (
                float(signal.starting_time)
                +
                np.arange(
                    number_of_samples
                )
                /
                float(signal.rate)
            )


        raise ValueError(
            f"Could not determine timestamps "
            f"for '{signal_name}'."
        )


    # ==========================================
    # GET COMPLETE SIGNAL
    # ==========================================

    def get_signal(
        self,
        signal_name
    ):

        data = self.get_signal_data(
            signal_name
        )

        timestamps = (
            self.get_signal_timestamps(
                signal_name
            )
        )

        if len(data) != len(timestamps):

            raise ValueError(
                f"Signal '{signal_name}' has "
                f"{len(data)} data samples but "
                f"{len(timestamps)} timestamps."
            )

        return {
            "data": data,
            "timestamps": timestamps
        }
