import pandas as pd
import numpy as np
import json
import re



def construct_b_matrix(pixel_ph_baseline_df: pd.DataFrame, quabo_uid: str) -> np.ndarray:
    """
    Constructs the 16x16 pixel baseline matrix (B_q) for a single Quabo.

    Args:
        pixel_ph_baseline_df (pd.DataFrame): DataFrame of pixel baselines.
        quabo_uid (str): The UID of the target Quabo.

    Returns:
        np.ndarray: A 16x16 NumPy array of baseline ADC values.
    """
    quabo_baselines = pixel_ph_baseline_df[pixel_ph_baseline_df['quabo_uid'] == quabo_uid]
    if quabo_baselines.empty:
        raise ValueError(f"No baseline data found for quabo_uid: {quabo_uid}")
    if len(quabo_baselines) != 256:
        raise ValueError(f"Expected 256 baseline values for {quabo_uid}, but found {len(quabo_baselines)}")

    # Sort by the original index to ensure correct ordering
    sorted_baselines = quabo_baselines.sort_values('coefs_idx')

    return sorted_baselines['baseline_adc'].values.reshape((16, 16))


def construct_g_matrix(pixel_ph_gain_delta_df: pd.DataFrame, quabo_uid: str) -> np.ndarray:
    """
    Constructs the 16x16 pixel gain delta matrix (G_q) for a single Quabo.

    Args:
        pixel_ph_gain_delta_df (pd.DataFrame): DataFrame of pixel gain deltas.
        quabo_uid (str): The UID of the target Quabo.

    Returns:
        np.ndarray: A 16x16 NumPy array of gain delta values.
    """
    quabo_gains = pixel_ph_gain_delta_df[pixel_ph_gain_delta_df['quabo_uid'] == quabo_uid]
    if quabo_gains.empty:
        raise ValueError(f"No gain delta data found for quabo_uid: {quabo_uid}")
    if len(quabo_gains) != 256:
        raise ValueError(f"Expected 256 gain delta values for {quabo_uid}, but found {len(quabo_gains)}")

    # Pivot the flat data back into a 2D matrix structure
    gain_pivot = quabo_gains.pivot(index='pixel_gain_key', columns='pixel_gain_idx', values='gain_delta')

    return gain_pivot.values


def _construct_block_matrix(detector_ph_calibration_df: pd.DataFrame, quabo_uid: str, column: str) -> np.ndarray:
    """Helper function to construct a 16x16 block matrix from quadrant-level data."""
    quabo_calib = detector_ph_calibration_df[detector_ph_calibration_df['quabo_uid'] == quabo_uid]
    if len(quabo_calib) != 4:
        raise ValueError(f"Expected 4 quadrant calibration entries for {quabo_uid}, but found {len(quabo_calib)}")

    # Create a map from quadrant number to the value
    value_map = pd.Series(quabo_calib[column].values, index=quabo_calib['detector_quadrant'])

    matrix = np.zeros((16, 16))

    # Fill each 8x8 quadrant with the corresponding value
    matrix[0:8, 0:8] = value_map[0]  # Quadrant 0 (top-left)
    matrix[0:8, 8:16] = value_map[1]  # Quadrant 1 (top-right)
    matrix[8:16, 0:8] = value_map[2]  # Quadrant 2 (bottom-left)
    matrix[8:16, 8:16] = value_map[3]  # Quadrant 3 (bottom-right)

    return matrix


def construct_n_matrix(detector_ph_calibration_df: pd.DataFrame, quabo_uid: str) -> np.ndarray:
    """
    Constructs the 16x16 'n' coefficient block matrix (N_q) for a single Quabo.

    Args:
        detector_ph_calibration_df (pd.DataFrame): DF of detector calibrations.
        quabo_uid (str): The UID of the target Quabo.

    Returns:
        np.ndarray: A 16x16 NumPy array of 'n' coefficients.
    """
    return _construct_block_matrix(detector_ph_calibration_df, quabo_uid, 'n')


def construct_m_matrix(detector_ph_calibration_df: pd.DataFrame, quabo_uid: str) -> np.ndarray:
    """
    Constructs the 16x16 'm' coefficient block matrix (M_q) for a single Quabo.

    Args:
        detector_ph_calibration_df (pd.DataFrame): DF of detector calibrations.
        quabo_uid (str): The UID of the target Quabo.

    Returns:
        np.ndarray: A 16x16 NumPy array of 'm' coefficients.
    """
    return _construct_block_matrix(detector_ph_calibration_df, quabo_uid, 'm')


def convert_adc_to_pe(I_q: np.ndarray, B_q: np.ndarray, G_q: np.ndarray, N_q: np.ndarray,
                      M_q: np.ndarray) -> np.ndarray:
    """
    Applies the full ADC to P.E. transformation for a single image frame.

    Args:
        I_q (np.ndarray): The 16x16 raw image data from a `ph256` frame.
        B_q (np.ndarray): The 16x16 baseline matrix.
        G_q (np.ndarray): The 16x16 gain delta matrix.
        N_q (np.ndarray): The 16x16 'n' coefficient matrix.
        M_q (np.ndarray): The 16x16 'm' coefficient matrix.

    Returns:
        np.ndarray: The 16x16 image data converted to P.E. units.
    """
    numerator = (I_q + B_q) - N_q
    denominator = M_q * G_q

    # Handle potential division by zero safely
    with np.errstate(divide='ignore', invalid='ignore'):
        pe_image = np.true_divide(numerator, denominator)
        pe_image[denominator == 0] = np.nan  # Set result to NaN where denominator is 0

    return pe_image
