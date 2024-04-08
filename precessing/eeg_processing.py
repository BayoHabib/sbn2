import logging
import mne
from mne.preprocessing import ICA
from autoreject import Ransac, get_rejection_threshold
from mne_icalabel import label_components

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
    - l_freq (float): The lower frequency boundary for the band-pass filter in Hz.
    - h_freq (float): The upper frequency boundary for the band-pass filter in Hz.
    - artifact_labels (list): A list of strings representing the labels of ICA components
      considered as artifacts (e.g., 'eye', 'muscle').

    Usage:
    ```python
    raw_eeg_data = mne.io.read_raw_fif('path/to/your/data.fif', preload=True)
    cleaner = EEGCleaner(raw_eeg_data, l_freq=1.0, h_freq=40.0)
    cleaner.fit()
    cleaned_data = cleaner.get_clean_data()
    ```
    """

    def __init__(self, raw_eeg_data, l_freq=1.0, h_freq=40.0, artifact_labels=None):
        if not isinstance(raw_eeg_data, mne.io.Raw):
            raise ValueError("raw_eeg_data must be an instance of mne.io.Raw")
        self.raw = raw_eeg_data.copy()
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.artifact_labels = artifact_labels if artifact_labels else ['eye', 'heart', 'muscle', 'line_noise', 'channel_noise', 'other']

    def fit(self):
        """
        Executes the artifact removal workflow on the EEG data. This method sequentially
        applies band-pass filtering, bad channel detection and interpolation, and ICA for
        artifact correction.
        """
        logger.info("Starting EEG data preprocessing...")
        self.filter_data()
        self.detect_and_interpolate_bad_channels()
        self.apply_ica()

    def filter_data(self):
        """
        Applies a band-pass filter to the EEG data to remove frequencies outside the
        specified range.
        """
        logger.info("Applying band-pass filter...")
        self.raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design='firwin')

    def detect_and_interpolate_bad_channels(self):
        """
        Detects bad channels using the RANSAC algorithm and interpolates them to improve
        data quality.
        """
        logger.info("Detecting and interpolating bad channels with RANSAC...")
        ransac = Ransac(verbose=True)
        self.raw = ransac.fit_transform(self.raw)

    def apply_ica(self):
        """
        Applies Independent Component Analysis (ICA) to the filtered EEG data for artifact
        correction. Components classified as artifacts based on the provided labels are
        excluded.
        """
        logger.info("Applying ICA for artifact correction...")
        ica = ICA(n_components=0.95, random_state=97, method='fastica')
        ica.fit(self.raw)
        labels = label_components(self.raw, ica)
        artifact_indices = self.identify_artifact_components(labels)
        ica.exclude = artifact_indices
        ica.apply(self.raw)

    def identify_artifact_components(self, labels):
        """
        Identifies components as artifacts based on the labels provided by ICLabel.

        Parameters:
        - labels (dict): The labels assigned to each component by ICLabel.

        Returns:
        - list: Indices of components identified as artifacts.
        """
        artifact_indices = [i for i, label in enumerate(labels['labels']) if label in self.artifact_labels]
        return artifact_indices

    def get_clean_data(self):
        """
        Retrieves the cleaned EEG data after preprocessing.

        Returns:
        - mne.io.Raw: The cleaned raw EEG data.
        """
        return self.raw
