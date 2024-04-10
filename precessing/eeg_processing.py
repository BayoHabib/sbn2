from typing import List
import mne
import logging
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep.prep_pipeline import PrepPipeline


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Define log formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Create file handler and set level and formatter
file_handler = logging.FileHandler('processor_log_file.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
# Create stream handler and set level
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class EEGCleaner:
    """
    EEGCleaner facilitates the preprocessing of EEG data by applying a series of artifact
    removal steps, including band-pass filtering, bad channel detection and interpolation,
    and Independent Component Analysis (ICA) for artifact correction.

    The class utilizes MNE-Python for EEG data handling, autoreject for automatic bad epoch
    rejection, and ICLabel for classifying ICA components to identify artifacts.

    Parameters:
    - raw_eeg_data (mne.io.Raw): The raw EEG data to be cleaned.
    - l_freq (float): The lower frequency boundary for the band-pass filter in Hz. Default: 0.1.
    - h_freq (float): The upper frequency boundary for the band-pass filter in Hz. Default: 40.0.
    - exclusion_threshold (float): The threshold for excluding ICA components based on artifact probability. Default: 0.8.
    - ransac (bool): Whether to use RANSAC for bad channel detection and interpolation. Default: True.

    Usage:
    ```python
    raw_eeg_data = mne.io.read_raw_fif('path/to/your/data.fif', preload=True)
    cleaner = EEGCleaner(raw_eeg_data, l_freq=1.0, h_freq=40.0)
    cleaner.fit()
    cleaned_data = cleaner.get_clean_data()
    ```
    """

    def __init__(self, raw_eeg_data: mne.io.Raw, l_freq: float = 0.1, h_freq: float = 40.0, exclusion_threshold: float = 0.8, ransac: bool = True) -> None:
        self.__processed_raw = raw_eeg_data.copy()
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.exclusion_threshold = exclusion_threshold
        self.ransac = ransac
        self.__noisy_channels = None

    def fit(self) -> None:
        """
        Executes the artifact removal workflow on the EEG data. This method sequentially
        applies band-pass filtering, bad channel detection and interpolation, and ICA for
        artifact correction.
        """
        logger.info("Starting EEG data preprocessing...")
        self.prep_pipeline()
        logger.info("Applying band-pass filter from 1 to 100 Hz for ICA preparation...")
        #self.__processed_raw.filter(l_freq=1., h_freq=100., fir_design='firwin')
        self.apply_ica()
        self.__processed_raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design='firwin')

    def prep_pipeline(self) -> None:
        # Setup for PyPREP - Ensure the data is loaded and filtered
        prep_params = {
            'ref_chs': 'eeg',
            'reref_chs': 'eeg',
            'line_freqs': np.arange(50, self.__processed_raw.info['sfreq']/2, 50),
        }
        if self.ransac:
            prep_params['ransac'] = True
        # Instantiate and run the PrepPipeline
        prep = PrepPipeline(self.__processed_raw, prep_params,montage = self.__processed_raw.get_montage())
        prep.fit()
        # Retrieve the processed data
        self.raw = prep.raw
        if not prep.still_noisy_channels:
            logger.info(f"List of still noisy channels: {prep.still_noisy_channels}")
            self.__noisy_channels = prep.still_noisy_channels

    def apply_ica(self) -> None:
        """
        Applies Independent Component Analysis (ICA) to the filtered EEG data for artifact
        correction. Components classified as artifacts based on the provided labels are
        excluded.
        """
        # Copy raw data to keep the original data intact
        tmp_raw = self.__processed_raw.copy()
        # Set EEG reference to average to minimize reference influence
        logger.info("Setting EEG reference to average.")
        tmp_raw.set_eeg_reference("average")
        logger.info("Applying band-pass filter from 1 to 100 Hz for ICA preparation...")
        tmp_raw.filter(l_freq=1., h_freq=100., fir_design='firwin')
        logger.info("Fitting ICA for artifact correction...")
        ica = ICA(n_components=0.95, random_state=97, method='infomax', fit_params=dict(extended=True))
        ica.fit(tmp_raw)
        # Automatically label the ICA components as artifacts or brain signals
        logger.info("Labeling ICA components...")
        ic_labels = label_components(tmp_raw, ica, method='iclabel')
        labels = ic_labels["labels"]
        probas = ic_labels["y_pred_proba"]
        # Identify artifact components based on labels and probabilities
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        # Format information about excluded components for logging
        exclude_info = "\n".join([
            f"- Component {idx}: Label = {labels[idx]}, Probability = {probas[idx]:.2f}"
            for idx in exclude_idx
        ])
        
        # Check if any component exceeds the exclusion threshold for being considered an artifact
        if np.any(ic_labels['y_pred_proba'][exclude_idx] > self.exclusion_threshold):
            logger.info(f"Excluding ICA components based on labels and probabilities:\n{exclude_info}")
            # Apply ICA exclusion only if at least one component exceeds the artifact probability threshold
            self.__processed_raw = tmp_raw.copy()
            ica.apply(self.__processed_raw, exclude=exclude_idx)
        else:
            logger.info("No ICA components exceeded the artifact probability threshold for exclusion.")  
        del tmp_raw  # Explicitly delete the temporary raw object to free up memory

    def get_clean_data(self) -> mne.io.Raw:
        """
        Retrieves the cleaned EEG data after preprocessing.

        Returns:
        - mne.io.Raw: The cleaned raw EEG data.
        """
        if self.__processed_raw is not None:
            return self.__processed_raw.copy()
        else:
            return None

    def get_noisy_channels(self) -> List[str]:
        """
        Retrieves the list of channels that were identified as noisy after preprocessing.

        Returns
        -------
        list of str
            List of channel names identified as noisy.
        """
        return self.__noisy_channels.copy()

    def __repr__(self) -> str:
        return f"EEGCleaner(raw_eeg_data={self.__processed_raw}, exclusion_threshold={self.exclusion_threshold}, l_freq={self.l_freq}, h_freq={self.h_freq}, ransac={self.ransac})"

    def __str__(self) -> str:
        return f"EEGCleaner(l_freq={self.l_freq}, h_freq={self.h_freq}, exclusion_threshold={self.exclusion_threshold}, ransac={self.ransac})"

    def fit(self) -> 'EEGCleaner':
        ...
        return self


# One should be discarded but keeping both for now
class EEGCleanerV2:
    """
    Facilitates the preprocessing of EEG data through a comprehensive pipeline including
    artifact removal techniques such as band-pass filtering, noisy channel detection with
    optional RANSAC, interpolation, and Independent Component Analysis (ICA) for fine-grained
    artifact correction. The class utilizes PyPREP for robust EEG preprocessing, complemented
    by MNE-Python for additional data handling and processing steps.

    Parameters
    ----------
    raw_eeg_data : mne.io.Raw
        The raw EEG data to be cleaned.
    l_freq : float
        Lower frequency boundary for the band-pass filter in Hz.
    h_freq : float
        Upper frequency boundary for the band-pass filter in Hz.
    exclusion_threshold : float, optional
        Threshold for excluding ICA components based on the probability of them being artifacts.
    ransac : bool, optional
        Flag to indicate whether RANSAC should be used for bad channel detection.

    Attributes
    ----------
    __processed_raw : mne.io.Raw
        The EEG data after preprocessing steps have been applied.
    __noisy_channels : list of str
        List of channels identified as noisy after preprocessing.

    Examples
    --------
    >>> raw_eeg_data = mne.io.read_raw_fif('path/to/your/data.fif', preload=True)
    >>> cleaner = EEGCleaner(raw_eeg_data, l_freq=1.0, h_freq=40.0)
    >>> cleaner.fit()
    >>> cleaned_data = cleaner.get_clean_data()
    """

    def __init__(self, raw_eeg_data, l_freq=1.0, h_freq=40.0, exclusion_threshold=0.8, ransac=True):
        self.__processed_raw = raw_eeg_data.copy()
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.exclusion_threshold = exclusion_threshold
        self.ransac = ransac
        self.__noisy_channels = None

    def fit(self):
        """
        Executes the preprocessing pipeline on the EEG data, applying a series of artifact
        removal steps in sequence.
        """
        logger.info("Starting EEG data preprocessing...")
        self.prep_pipeline()
        self.apply_ica()
        self.__processed_raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design='firwin')
        logger.info("EEG data preprocessing completed.")

    def prep_pipeline(self):
        """
        Applies PyPREP's PrepPipeline to perform robust preprocessing including noisy channel
        detection, interpolation, and referencing. RANSAC can be optionally applied for enhanced
        noisy channel detection.
        """
        logger.info("Applying PyPREP pipeline for initial preprocessing...")
        prep_params = {
            'ref_chs': 'eeg',
            'reref_chs': 'eeg',
            'line_freqs': np.arange(50, self.__processed_raw.info['sfreq'] / 2, 50),
        }
        if self.ransac:
            prep_params['ransac'] = True
        prep = PrepPipeline(self.__processed_raw, prep_params,montage = self.__processed_raw.get_montage())
        prep.fit()
        self.__processed_raw = prep.raw
        if prep.still_noisy_channels:
            self.__noisy_channels = prep.still_noisy_channels
            logger.info(f"List of still noisy channels: {prep.still_noisy_channels}")

    def apply_ica(self):
        """
        Applies ICA to identify and exclude components associated with artifacts. Components
        are automatically labeled, and those identified as artifacts based on the exclusion
        threshold are removed from the data.
        """
        logger.info("Fitting ICA for artifact correction...")
        ica = ICA(n_components=0.95, random_state=97, method='infomax', fit_params={'extended': True})
        ica.fit(self.__processed_raw)
        logger.info("Labeling ICA components...")
        ic_labels = label_components(self.__processed_raw, ica, method='iclabel')
        exclude_idx = [idx for idx, label in enumerate(ic_labels["labels"]) if label not in ["brain", "other"]]
        if np.any(ic_labels['y_pred_proba'][exclude_idx] > self.exclusion_threshold):
            logger.info(f"Excluding ICA components based on labels and probabilities.")
            ica.apply(self.__processed_raw, exclude=exclude_idx)

    def get_clean_data(self):
        """
        Retrieves the cleaned EEG data after preprocessing.

        Returns
        -------
        mne.io.Raw
            The cleaned raw EEG data.
        """
        return self.__processed_raw

    def get_noisy_channels(self):
        """
        Retrieves the list of channels that were identified as noisy after preprocessing.
    
        Returns
        -------
        list of str
            List of channel names identified as noisy.
        """
        return self.__noisy_channels

