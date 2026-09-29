
# ==========================================
# CONTINUOUS NEURAL DECODER
#
# Decodes:
#
#     neural population activity
#                 ↓
#         finger velocity
#
# Input:
#
#     X = neural activity
#
# Output:
#
#     y = [X velocity, Y velocity]
#
# Model:
#
#     Ridge regression
#
# Evaluation:
#
#     R²
#     RMSE
#     MAE
#     Pearson correlation
#
# ==========================================


# ==========================================
# IMPORTS
# ==========================================

import numpy as np

from sklearn.linear_model import Ridge


# ==========================================
# NEURAL DECODER
# ==========================================

class NeuralDecoder:

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(
        self,
        test_size=0.20,
        alpha=1.0
    ):

        self.test_size = float(
            test_size
        )

        self.alpha = float(
            alpha
        )


        # --------------------------------------
        # MODEL
        # --------------------------------------

        self.model = Ridge(
            alpha=self.alpha
        )


        # --------------------------------------
        # TRAINING STATE
        # --------------------------------------

        self.is_trained = False


        # --------------------------------------
        # DATA
        # --------------------------------------

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None


        # --------------------------------------
        # PREDICTIONS
        # --------------------------------------

        self.predictions = None


        # --------------------------------------
        # METRICS
        # --------------------------------------

        self.r2 = None
        self.rmse = None
        self.mae = None

        self.correlation_x = None
        self.correlation_y = None


    # ==========================================
    # PREPARE DATA
    # ==========================================
    #
    # Chronological split.
    #
    # We do NOT randomly shuffle the neural
    # time series.
    #
    # Example:
    #
    # 0% ─────────────── 80% ────────── 100%
    #       TRAIN              TEST
    #
    # ==========================================

    def prepare_data(
        self,
        features,
        targets
    ):

        X = np.asarray(
            features,
            dtype=float
        )

        y = np.asarray(
            targets,
            dtype=float
        )


        # --------------------------------------
        # VALIDATE DIMENSIONS
        # --------------------------------------

        if X.ndim != 2:

            raise ValueError(
                "Features must be a 2-dimensional array."
            )


        if y.ndim != 2:

            raise ValueError(
                "Targets must be a 2-dimensional array."
            )


        if len(X) != len(y):

            raise ValueError(
                "Number of feature samples must "
                "match number of target samples."
            )


        if len(X) < 2:

            raise ValueError(
                "At least two samples are required."
            )


        # --------------------------------------
        # VALIDATE TEST SIZE
        # --------------------------------------

        if not 0.0 < self.test_size < 1.0:

            raise ValueError(
                "test_size must be between 0 and 1."
            )


        # --------------------------------------
        # DETERMINE SPLIT POINT
        # --------------------------------------

        split_index = int(
            len(X)
            *
            (1.0 - self.test_size)
        )


        if split_index <= 0:

            raise ValueError(
                "Training set would be empty."
            )


        if split_index >= len(X):

            raise ValueError(
                "Testing set would be empty."
            )


        # ======================================
        # CHRONOLOGICAL SPLIT
        # ======================================

        self.X_train = X[
            :split_index
        ]


        self.X_test = X[
            split_index:
        ]


        self.y_train = y[
            :split_index
        ]


        self.y_test = y[
            split_index:
        ]


        return (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        )


    # ==========================================
    # TRAIN
    # ==========================================

    def train(self):

        if self.X_train is None:

            raise ValueError(
                "Training data has not been prepared."
            )


        self.model.fit(
            self.X_train,
            self.y_train
        )


        self.is_trained = True


    # ==========================================
    # PREDICT
    # ==========================================

    def predict(
        self,
        features
    ):

        if not self.is_trained:

            raise RuntimeError(
                "Decoder has not been trained."
            )


        X = np.asarray(
            features,
            dtype=float
        )


        return self.model.predict(X)


    # ==========================================
    # ROOT MEAN SQUARED ERROR
    # ==========================================

    def _calculate_rmse(
        self,
        actual,
        predicted
    ):

        error = (
            actual
            -
            predicted
        )


        mse = np.mean(
            error ** 2,
            axis=0
        )


        rmse = np.sqrt(
            mse
        )


        return rmse


    # ==========================================
    # MEAN ABSOLUTE ERROR
    # ==========================================

    def _calculate_mae(
        self,
        actual,
        predicted
    ):

        absolute_error = np.abs(
            actual
            -
            predicted
        )


        return np.mean(
            absolute_error,
            axis=0
        )


    # ==========================================
    # R-SQUARED
    # ==========================================

    def _calculate_r2(
        self,
        actual,
        predicted
    ):

        residual = (
            actual
            -
            predicted
        )


        residual_sum = np.sum(
            residual ** 2,
            axis=0
        )


        centered = (
            actual
            -
            np.mean(
                actual,
                axis=0
            )
        )


        total_sum = np.sum(
            centered ** 2,
            axis=0
        )


        # Avoid division by zero.

        r2 = np.zeros(
            actual.shape[1],
            dtype=float
        )


        valid = (
            total_sum > 0
        )


        r2[valid] = (
            1.0
            -
            (
                residual_sum[valid]
                /
                total_sum[valid]
            )
        )


        return r2


    # ==========================================
    # PEARSON CORRELATION
    # ==========================================

    def _calculate_correlation(
        self,
        actual,
        predicted
    ):

        correlations = []


        for dimension in range(
            actual.shape[1]
        ):

            actual_dimension = (
                actual[:, dimension]
            )


            predicted_dimension = (
                predicted[:, dimension]
            )


            actual_std = np.std(
                actual_dimension
            )


            predicted_std = np.std(
                predicted_dimension
            )


            if (
                actual_std == 0
                or
                predicted_std == 0
            ):

                correlation = np.nan

            else:

                correlation = np.corrcoef(
                    actual_dimension,
                    predicted_dimension
                )[0, 1]


            correlations.append(
                correlation
            )


        return np.asarray(
            correlations,
            dtype=float
        )


    # ==========================================
    # EVALUATE
    # ==========================================

    def evaluate(self):

        if not self.is_trained:

            raise RuntimeError(
                "Decoder has not been trained."
            )


        # --------------------------------------
        # PREDICT TEST DATA
        # --------------------------------------

        self.predictions = self.predict(
            self.X_test
        )


        # --------------------------------------
        # CALCULATE METRICS
        # --------------------------------------

        self.r2 = self._calculate_r2(
            self.y_test,
            self.predictions
        )


        self.rmse = self._calculate_rmse(
            self.y_test,
            self.predictions
        )


        self.mae = self._calculate_mae(
            self.y_test,
            self.predictions
        )


        self.correlation = (
            self._calculate_correlation(
                self.y_test,
                self.predictions
            )
        )


        self.correlation_x = (
            self.correlation[0]
        )


        self.correlation_y = (
            self.correlation[1]
        )


        # --------------------------------------
        # RETURN RESULTS
        # --------------------------------------

        return {

            "r2": self.r2,

            "rmse": self.rmse,

            "mae": self.mae,

            "correlation_x":
                self.correlation_x,

            "correlation_y":
                self.correlation_y
        }


    # ==========================================
    # GET PREDICTIONS
    # ==========================================

    def get_predictions(self):

        if self.predictions is None:

            raise RuntimeError(
                "Decoder has not been evaluated."
            )


        return self.predictions


    # ==========================================
    # GET METRICS
    # ==========================================

    def get_metrics(self):

        if self.predictions is None:

            raise RuntimeError(
                "Decoder has not been evaluated."
            )


        return {

            "r2":
                self.r2,

            "rmse":
                self.rmse,

            "mae":
                self.mae,

            "correlation_x":
                self.correlation_x,

            "correlation_y":
                self.correlation_y
        }
