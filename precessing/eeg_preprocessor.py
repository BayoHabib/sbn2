import mne
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('my_log_file.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class EEGPreprocessor:
    """
    A class for preprocessing EEG data.

    Attributes:
        raw_file_path (str): Path to the raw EEG file.
        output_file (str): Path to the output file.
        montage_system (str): Montage system to be used.
        misc_channels (list): List of miscellaneous channels to be removed.

    Methods:
        load_eeg(): Load the raw EEG data.
        save_processed_file(): Save the processed EEG data.
        get_bad_channels(): Get the list of bad channels.
        set_bad_channels(bad_channels): Set the list of bad channels.
        remove_artifacts(): Remove artifacts from the EEG data.
        set_montage(): Set the montage for the EEG data.
        get_artifacts(): Get the list of artifacts.
    """

    def __init__(self, raw_file_path, output_file, montage_system='standard_1020', misc_channels=None):
        """
        Initialize the EEGPreprocessor class.

        Parameters:
            raw_file_path (str): Path to the raw EEG file.
            output_file (str): Path to the output file.
            montage_system (str): Montage system to be used.
            misc_channels (list): List of miscellaneous channels to be removed.
        """
        self.raw_file_path = raw_file_path
        self.output_file = output_file
        self.montage_system = montage_system
        self.misc_channels = misc_channels
        self.__raw = None
        self.__preprocessed_signal = None
    def load_eeg(self):
        """
        Load the raw EEG data.
        """
        logger.info(f"Loading raw EEG data from {self.raw_file_path}")
        
        self.__raw = mne.io.read_raw(self.raw_file_path, preload=True)
            
    def save_processed_file(self):
        """
        Save the processed EEG data.
        """
        pass
        #logger.info(f"Saving processed EEG data to {self.output_file}")
        #self.raw.save(self.output_file, overwrite=True)

    def set_channel_names(self, channel_mapping):
        """
        Map the names of the channels in the raw dataset to new names.

        Parameters:     
            channel_mapping (dict): A dictionary  mapping old channel names  to new channel names.
                                     The keys are the original channel names and the values are the new channel names.
        """
        logger.info('Mapping Channel Names')
        if not isinstance(channel_mapping, dict):
            raise ValueError('Channel mapping must be a dictionary')
        
        try:
            self.__raw.rename_channels(channel_mapping)
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
            self.__raw.set_montage(montage)
        except Exception as e:
            print(e)
            raise ValueError("Invalid channels name for the system")

    def set_channel_types(self, channel_type_dict):
        """
        Define the types of each channel in the EEG data.

        Parameters:
        channel_type_dict (dict): A dictionary containing the channel names as keys and the corresponding channel type as value.
        """
        logger.info('Defining channel types')
        if not isinstance(channel_type_dict, dict):
            raise ValueError('Channel mapping must be a dictionary')
        try:
            self.__raw.set_channel_types(channel_type_dict, verbose=None)
        except Exception as e:
            print(e)
            raise ValueError('The provided type mapping does not match the data')
                                  
    def get_raw(self):
        return self.__raw
    