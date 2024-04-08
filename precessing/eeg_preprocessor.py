import os
import mne
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define log formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create file handler and set level and formatter
file_handler = logging.FileHandler('my_log_file.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create stream handler and set level
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class EEGPreprocessor:
    """
    A class for preprocessing EEG data.

    Attributes:
        raw_file_path (str): Path to the raw EEG file.
        output_file (str): Path to the output file.
        montage_system (str): Montage system to be used.
        misc_channels (dict): List of miscellaneous channels to be removed.
        channel_mapping (dict): A dictionary mapping old channel names to new channel names.
                                The keys are the original channel names and the values are the new channel names.
        channel_types (dict): A dictionary containing the channel names as keys and the corresponding channel type as value.

    Methods:
        load_eeg(): Load the raw EEG data.
        save_processed_file(): Save the processed EEG data.
        set_montage(): Set the montage for the EEG data.
        set_channel_names(): Map the names of the channels in the raw dataset to new names.
        set_channel_types(): Define the types of each channel in the EEG data.
        get_raw(): Return the raw EEG data.
        get_preprocessed(): Return the preprocessed EEG data.
        fit(): Placeholder method for model fitting (not implemented).
    """

    def __init__(self, raw_file_path, output_dir, channel_types=None, channel_mapping=None,
                 montage_system='standard_1020', misc_channels=None):
        """
        Initialize the EEGPreprocessor class.

        Parameters:
            raw_file_path (str): Path to the raw EEG file.
            output_file (str): Path to the output file.
            montage_system (str): Montage system to be used.
            misc_channels (list): List of miscellaneous channels to be removed.
        """
        self.raw_file_path = raw_file_path
        self.output_dir = output_dir
        self.montage_system = montage_system
        self.misc_channels = misc_channels
        self.__raw = None
        self.__preprocessed_signal = None
        self.channel_mapping = channel_mapping
        self.channel_types = channel_types

    def load_eeg(self):
        """
        Load the raw EEG data.
        """
        logger.info(f"Loading raw EEG data from {self.raw_file_path}")
        self.__raw = mne.io.read_raw(self.raw_file_path, preload=True)

    def save_processed_file(self):
        """
        Save the preprocessed EEG data.
        """
        if self.__preprocessed_signal is None:
            raise ValueError("No preprocessed data to save. Please preprocess the data first.")

        logger.info(f"Saving processed EEG data to {self.output_file}")
        filename = f"{os.path.basename(self.raw_file_path).split('.')[0]}.fif"
        file_path = os.path.join(self.output_dir, filename)
        self.__preprocessed_signal.save(file_path, overwrite=True)


    def set_channel_names(self):
        """
        Map the names of the channels in the raw dataset to new names.
        """
        logger.info('Mapping Channel Names')
        if not isinstance(self.channel_mapping, dict):
            raise ValueError('Channel mapping must be a dictionary')

        try:
            self.__preprocessed_signal.rename_channels(self.channel_mapping)
        except Exception as e:
            print(e)
            raise TypeError('Provided channel map does not match data')

    def set_montage(self):
        """
        Set the montage for the EEG data.
        """
        logger.info(f"Setting montage to {self.montage_system}")
        try:
            montage = mne.channels.make_standard_montage(self.montage_system)
            self.__preprocessed_signal.set_montage(montage)
        except Exception as e:
            print(e)
            raise ValueError("Invalid channels name for the system")

    def set_channel_types(self):
        """
        Define the types of each channel in the EEG data.
        """
        logger.info('Defining channel types')
        if not isinstance(self.channel_types, dict):
            raise ValueError('Channel mapping must be a dictionary')
        try:
            self.__preprocessed_signal.set_channel_types(self.channel_types, verbose=None)
        except Exception as e:
            print(e)
            raise ValueError('The provided type mapping does not match the data')

    def get_raw(self):
        """
        Returns the raw EEG data.
        """
        return self.__raw

    def get_preprocessed(self):
        """
        Returns the preprocessed EEG data.
        """
        return self.__preprocessed_signal

def fit(self):
    """
    Preprocess EEG data and save the processed file.

    This method executes the preprocessing steps including loading EEG data, setting channel names,
    setting channel types, setting montage, and saving the processed file.

    Raises:
        ValueError: If there is no preprocessed data to save.
    """
    try:
        # Load raw EEG data
        self.load_eeg()
        
        # Set channel names
        if self.channel_mapping:
            self.set_channel_names()

        # Set channel types
        if self.channel_types:
            self.set_channel_types()

        # Set montage
        self.set_montage()

        # Save processed file
        self.save_processed_file()
        
        logger.info("Preprocessing completed successfully.")
    except Exception as e:
        logger.error(f"Error occurred during preprocessing: {str(e)}")
        raise

