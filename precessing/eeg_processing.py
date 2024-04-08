def compute_psd():
    pass

def band_power():
    pass

def feature_extraction():

    pass

def remove_artifacts(self):
    """
    Remove artifacts from the EEG data.
    """
    pass
    #logger.info("Removing artifacts")
    # Implement artifact removal here
    #self.__preprocessed_signal = None # receive the processed signal after removing artifacts

def set_bad_channels(self, bad_channels):
    """
    Set the list of bad channels.

    Parameters:
        bad_channels (list): List of bad channels.
    """
    pass
    #logger.info(f"Setting bad channels to {bad_channels}")
    #self.raw.info['bads'] = bad_channels

def get_bad_channels(self):
        """
        Get the list of bad channels.
        """
        pass
        #logger.info("Getting bad channels")
        #return self.raw.info['bads']

def get_artifacts(self):
        """
        Get the list of artifacts.
        """
        pass
        #logger.info("Getting artifacts")
        # Implement artifact detection here