
# ==========================================
# INTRACORTICAL BCI DECODER
#
# NEURAL DATA EXTRACTION
#
# PURPOSE:
#
# Extract neuronal spike data from an NWB
# intracortical recording.
#
# INPUT:
#
#     NWBFile
#
# OUTPUT:
#
#     Unit IDs
#     Spike times
#     Number of neurons
#     Spike counts
#     Recording duration
#     Firing rates
# ==========================================

import numpy as np


class NeuralData:

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(self, nwbfile):

        self.nwbfile = nwbfile

        self.unit_ids = None

        self.spike_times = {}

        self.number_of_units = 0


    # ==========================================
    # EXTRACT NEURAL UNITS
    # ==========================================

    def extract_units(self):

        if self.nwbfile.units is None:

            raise ValueError(
                "The NWB file does not contain "
                "a neural unit table."
            )


        units = self.nwbfile.units


        # --------------------------------------
        # GET ACTUAL UNIT IDs
        # --------------------------------------

        self.unit_ids = np.asarray(
            units.id[:]
        )


        self.number_of_units = len(
            self.unit_ids
        )


        # --------------------------------------
        # EXTRACT SPIKE TIMES
        #
        # IMPORTANT:
        #
        # unit_id is NOT necessarily the same
        # as the row index in the NWB table.
        #
        # Therefore we enumerate the rows.
        # --------------------------------------

        self.spike_times = {}


        for unit_index, unit_id in enumerate(
            self.unit_ids
        ):

            spike_times = units[
                "spike_times"
            ][unit_index]


            self.spike_times[unit_id] = (
                np.asarray(
                    spike_times,
                    dtype=float
                )
            )


        return self.spike_times


    # ==========================================
    # GET SPIKE TIMES
    # ==========================================

    def get_spike_times(
        self,
        unit_id
    ):

        if not self.spike_times:

            self.extract_units()


        if unit_id not in self.spike_times:

            raise KeyError(
                f"Unit {unit_id} was not found."
            )


        return self.spike_times[unit_id]


    # ==========================================
    # GET UNIT IDs
    # ==========================================

    def get_unit_ids(self):

        if self.unit_ids is None:

            self.extract_units()


        return self.unit_ids


    # ==========================================
    # GET NUMBER OF UNITS
    # ==========================================

    def get_number_of_units(self):

        if self.unit_ids is None:

            self.extract_units()


        return self.number_of_units


    # ==========================================
    # GET SPIKE COUNT
    # ==========================================

    def get_spike_count(
        self,
        unit_id
    ):

        spike_times = self.get_spike_times(
            unit_id
        )


        return len(spike_times)


    # ==========================================
    # GET RECORDING DURATION
    # ==========================================

    def get_recording_duration(self):

        if not self.spike_times:

            self.extract_units()


        earliest_spike = None

        latest_spike = None


        for spike_times in (
            self.spike_times.values()
        ):

            if len(spike_times) == 0:

                continue


            current_earliest = float(
                np.min(spike_times)
            )


            current_latest = float(
                np.max(spike_times)
            )


            if (
                earliest_spike is None
                or
                current_earliest < earliest_spike
            ):

                earliest_spike = (
                    current_earliest
                )


            if (
                latest_spike is None
                or
                current_latest > latest_spike
            ):

                latest_spike = (
                    current_latest
                )


        if (
            earliest_spike is None
            or
            latest_spike is None
        ):

            return 0.0


        return float(
            latest_spike
            -
            earliest_spike
        )


    # ==========================================
    # GET FIRING RATE
    # ==========================================

    def get_firing_rate(
        self,
        unit_id
    ):

        spike_times = self.get_spike_times(
            unit_id
        )


        if len(spike_times) < 2:

            return 0.0


        duration = (
            float(np.max(spike_times))
            -
            float(np.min(spike_times))
        )


        if duration <= 0:

            return 0.0


        firing_rate = (
            len(spike_times)
            /
            duration
        )


        return float(
            firing_rate
        )
