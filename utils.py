import mne
import os
import config
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann
import logging
import utils
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_config = config.EEGAnalysisConfig()

def extract_filename_without_extension(file_path):
    # Extract the base name of the file (e.g., 'ale.edf' from the full path)
    base_name = os.path.basename(file_path)
    # Split the base name by the last dot ('.') and take the first part to exclude the extension
    file_name_without_extension = os.path.splitext(base_name)[0]
    return file_name_without_extension

def eeg_to_epochs(eeg_file_path, l_freq=_config.low_freq, h_freq=_config.high_freq, 
                             epoch_duration = _config.epoch_duration, overlap=_config.overlap_duration):
    """
    Preprocess EEG data by filtering, removing artifacts, creating fixed-length epochs,
    and return epochs file.

    Parameters:
    - eeg_file_path: Path to the raw EEG file.
    - l_freq: Lower frequency bound for the band-pass filter (in Hz).
    - h_freq: Higher frequency bound for the band-pass filter (in Hz).
    - epoch_duration: Duration of each epoch in seconds.
    - overlap: Duration of overlap between consecutive epochs in seconds.
    """
    logger.info(f"Starting EEG preprocessing for file: {eeg_file_path}")
    logger.info(f"Filtering settings - Low frequency: {l_freq} Hz, High frequency: {h_freq} Hz")
    logger.info(f"Epoch settings - Duration: {epoch_duration}s, Overlap: {overlap}s")
    # Load the raw EEG data
    logger.info("Loading raw EEG data...")
    raw = mne.io.read_raw(eeg_file_path, preload=True)
    logger.info("Renaming channels according to standard...")
    raw.rename_channels(mapping=_config.new_channels_names)
    logger.info("Setting channel types...")
    raw.set_channel_types(mapping=_config.ch_types)
    logger.info("Setting montage to standard 10-20 system...")
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    # Step 1: Filtering
    logger.info("Applying band-pass filter...")
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    # Step 2: Create Fixed-Length Epochs
    logger.info("Creating fixed-length events for epoching...")
    events = mne.make_fixed_length_events(raw, start=0, stop=raw.times[-1], duration=epoch_duration - overlap)
    logger.info(f"Total events created: {len(events)}")
    epochs = mne.Epochs(raw, events, tmin=0.0, tmax=epoch_duration, baseline=None, preload=True)
    # Step 3: Drop Unwanted Channels
    logger.info("Dropping unwanted channels: EEG A2-A1, ECG ECG")
    epochs.drop_channels(['EEG A2-A1', 'ECG ECG'], on_missing='ignore')
    logger.info("Preprocessing complete. Epochs ready for analysis.")
    return epochs

elec_locs = np.array(_config.electrodes_locs)

def azimuthal_equidistant_projection(points):
    """
    Perform an azimuthal equidistant projection of 3D points onto a 2D plane.
    
    Parameters:
    - points: NumPy array of shape (n_channels, 3) representing 3D points (x, y, z).
    
    Returns:
    - A NumPy array of shape (n_channels, 2) with the 2D projected points.
    """
    # Calculate the distance from each point to the origin
    d = np.sqrt(np.sum(points**2, axis=1))
    
    # Avoid division by zero for the origin point
    d[d == 0] = np.finfo(float).eps
    
    # Calculate the angle theta from the z-axis and azimuth phi from the x-axis
    theta = np.arccos(points[:, 2] / d)
    phi = np.arctan2(points[:, 1], points[:, 0])
    
    # Perform the azimuthal equidistant projection
    x_proj = d * np.sin(theta) * np.cos(phi)
    y_proj = d * np.sin(theta) * np.sin(phi)
    
    # Combine the projected coordinates into a single array
    projected_points = np.vstack((x_proj, y_proj)).T
    
    return projected_points


def create_dict_from_names(names, N_epochs):
    """
    Creates a dictionary where each key is a name from the 'names' list,
    and each value is a list of integers, ensuring 'N_epochs' is divisible by the number of names.
    
    Parameters:
    - names: List of strings, each being a name.
    - N_epochs: Integer, the total count of epochs that must be divisible by the length of 'names'.
    
    Returns:
    A dictionary with names as keys and lists of integers as values.
    
    Raises:
    ValueError: If 'N_epochs' is not divisible by the length of 'names'.
    """
    logging.info("Starting create_dict_from_names with %d names and %d epochs", len(names), N_epochs)
    
    # Check if N_epochs is divisible by the number of names
    if N_epochs % len(names) != 0:
        logging.error("N_epochs (%d) is not divisible by the number of names (%d)", N_epochs, len(names))
        raise ValueError("N_epochs must be divisible by the number of names.")
    
    output_dict = {}
    start = 0
    increment = N_epochs // len(names)
    
    for name in names:
        output_dict[name] = list(range(start, start + increment))
        start += increment
        logging.debug("Generated list for %s: %s", name, output_dict[name])
    
    logging.info("Successfully created dictionary from names and epochs")
    return output_dict


class LstmCnnDataGenerator_v2(PyDataset):
    """Generates data for TensorFlow/Keras from multiple .npy files containing image and EEG data."""
    def __init__(self, image_paths, eeg_paths, labels, batch_size=32, shuffle=True,**kwargs):
        """Initialization of the data generator.
        Args:
            image_paths (list): List of paths to the .npy files containing images.
            eeg_paths (list): List of paths to the .npy files containing EEG signals.
            labels (list): List of labels for each sample.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data at the end of each epoch.
        """
        super().__init__(**kwargs)
        self.image_paths = image_paths
        self.eeg_paths = eeg_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Calculate cumulative lengths for indexing across multiple files
        self.cumulative_lengths = self._calculate_cumulative_lengths()
        self.indices = np.arange(len(labels))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data.
        This method fetches a batch of data by mapping global indices (across all files)
        to the corresponding file and local index within that file.
        """
        # Calculate the start and end index for the current batch
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, eegs, batch_labels = [], [], []
        
        for global_idx in batch_indices:
            # Map the global index to the corresponding file index and local sample index
            file_idx, sample_idx = self._find_file_and_sample_idx(global_idx)
            
            # Load the image and EEG data for the calculated indices
            image_path = self.image_paths[file_idx]
            eeg_path = self.eeg_paths[file_idx]
            image = np.load(image_path, mmap_mode='r')[sample_idx]
            eeg = np.load(eeg_path, mmap_mode='r')[sample_idx]
            
            images.append(image)
            eegs.append(eeg)
            batch_labels.append(self.labels[global_idx])
        return {'image_input': np.array(images), 'eeg_input': np.array(eegs).transpose(0,2,1)}, np.array(batch_labels)

    def on_epoch_end(self):
        """Shuffles indices after each epoch if shuffle is set to True."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _calculate_cumulative_lengths(self):
        """Calculates cumulative lengths of all files for easier indexing.
        This helps in determining how many samples have been processed and how to
        locate the next sample across multiple files.
        """
        lengths = [np.load(path, mmap_mode='r').shape[0] for path in self.image_paths]
        return np.cumsum(lengths)

    def _find_file_and_sample_idx(self, global_idx):
        """Finds which file and local index corresponds to the global index.
        This method maps a global index (across all samples in all files) to a specific file and
        a local index within that file.
        """
        file_idx = np.searchsorted(self.cumulative_lengths, global_idx, side='right')
        if file_idx == 0:
            sample_idx = global_idx  # If in the first file, global index is the local index
        else:
            sample_idx = global_idx - self.cumulative_lengths[file_idx - 1]  # Adjust index for previous files
        return file_idx, sample_idx
