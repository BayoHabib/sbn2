# %% [markdown]
# ## Relevant libraries

# %% [markdown]
# All the libraries here are required. Install them before launching the notebooks

# %%
import numpy as np
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
import pandas as pd
import utils # custom library
import mne
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import os
import matplotlib.pyplot as plt
import config # custom

# %%
config.EEGAnalysisConfig().epoch_duration

# %% [markdown]
# ## Some global variables

# %%
OUTPUT_DIR = 'C:\\Users\\Administrator\\Downloads\\EEG_images'
OUTPUT_DIR_EPOCHS = 'C:\\Users\\Administrator\\Downloads\\EEG_epochs'
OUTPUT_DIR_PREPROCESSED_DATA = 'C:\\Users\\Administrator\\EEG_preprocessed_data\\preprocessed_imgs.npy'        
AZ_PROJ_VALUE = np.array([
    [-0.0309026, 0.11458518],
    [0.02840949, 0.11534631],
    [-0.05180905, 0.0866879],
    [0.05027427, 0.08743839],
    [-0.07187663, 0.07310353],
    [0.07143526, 0.07450512],
    [-0.08598209, 0.01487164],
    [0.08326136, 0.01525818],
    [-0.06714873, 0.02335823],
    [0.06532887, 0.0235731],
    [-0.07445797, -0.04212316],
    [0.07103246, -0.04225998],
    [-0.05503824, -0.0442103],
    [0.05363601, -0.04433453],
    [-0.03157357, -0.08056835],
    [0.02768309, -0.08048884],
    [-0.00122928, 0.09327445],
    [-0.00137414, 0.02761709],
    [-0.00170945, -0.04521299]
])
BANDS_FREQ = {'alpha': (8, 12), 'beta': (13, 30), 'theta': (4, 7)}
NAME_BURNOUT_LEVEL_FILE = 'C:\\Users\\Administrator\\Downloads\\name_burnout.xlsx'
df = pd.read_excel(NAME_BURNOUT_LEVEL_FILE)
df_filtered = df.dropna(subset=['burnout_Boiko']) # dropping missing values
EEG_PATH_FILE = df_filtered['path_file']
IMG_RESOLUTION = 28
#n_gridpoints = config.IMG_RESOLUTION # image resolution here
#output_dir = config.OUTPUT_DIR # output dir of the generated images
#eeg_path_file = config.EEG_PATH_FILE
_config = config.EEGAnalysisConfig()

# %%
for file_path in EEG_PATH_FILE:
    print(file_path)
len(EEG_PATH_FILE)

# %% [markdown]
# ## Data preprocessing workflow

# %% [markdown]
# 
# ![Data preprocessing workflow](data_preprocessing.png)
# 

# %% [markdown]
# ## Features extraction for CNN

# %%
def extract_frequency_bands(file_path, freq_bands = BANDS_FREQ):
    """
    Extracts average power spectral density (PSD) in specified frequency bands for each channel
    in each epoch of an MNE Epochs object.

    Parameters:
    - file_path: raw file path containing the EEG signal.
    - freq_bands: Dictionary with frequency band names as keys and tuples (fmin, fmax) as values.

    Returns:
    - numpy.ndarray of shape [number_of_epochs, (n_channels*3)], where each element represents
      the average PSD in a frequency band for a channel in an epoch.
    """
    epochs = utils.eeg_to_epochs(file_path)
    # Initialize an empty list to store the results
    results = []

    # Iterate over each epoch
    for epoch in epochs:
        # Apply Hann window to minimize spectral leakage
        hann_window = np.hanning(epoch.shape[1])
        windowed_epoch = epoch * hann_window
        
        # Perform FFT and calculate power spectral density (PSD)
        fft_epoch = np.fft.rfft(windowed_epoch, axis=1)
        freqs = np.fft.rfftfreq(epoch.shape[1], d=1/epochs.info['sfreq'])
        psd_epoch = np.abs(fft_epoch) ** 2
        
        # Container for the current epoch's features
        epoch_features = []
        
        # Iterate over each channel
        for i_channel in range(len(epochs.ch_names)):
            # Extract and concatenate frequency bands
            for band in freq_bands.values():
                fmin, fmax = band
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                band_power = psd_epoch[i_channel, freq_mask].mean()
                epoch_features.append(band_power)
        
        # Append the features of the current epoch to the results list
        results.append(epoch_features)

    # Convert the list to a NumPy array
    return np.array(results)


# %% [markdown]
# ## Parallelization of features extraction

# %%
from joblib import Parallel, delayed
import numpy as np
import utils

def process_epoch(epoch, freq_bands, sfreq, index):
    """Process a single epoch and return the index with the results."""
    hann_window = np.hanning(epoch.shape[1])
    windowed_epoch = epoch * hann_window
    fft_epoch = np.fft.rfft(windowed_epoch, axis=1)
    freqs = np.fft.rfftfreq(epoch.shape[1], d=1/sfreq)
    psd_epoch = np.abs(fft_epoch)  # Note: Assuming you want PSD, not just amplitude

    epoch_features = []
    for i_channel in range(epoch.shape[0]):  # Assuming epoch shape is [channels, samples]
        for band in freq_bands.values():
            fmin, fmax = band
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = psd_epoch[i_channel, freq_mask].mean()
            epoch_features.append(band_power)

    return (index, epoch_features)

def extract_frequency_bands_par(file_path, freq_bands):
    epochs = utils.eeg_to_epochs(file_path)
    sfreq = epochs.info['sfreq']  # Sampling frequency
    
    # Process each epoch with its index
    results_with_indices = Parallel(n_jobs=-1)(
        delayed(process_epoch)(epoch, freq_bands, sfreq, i) for i, epoch in enumerate(epochs.get_data())
    )
    
    # Sort results by the original indices to preserve order
    sorted_results = sorted(results_with_indices, key=lambda x: x[0])
    
    # Extract the sorted results, discarding the indices
    sorted_features = [features for index, features in sorted_results]
    
    return np.array(sorted_features)


# %% [markdown]
# Let test the parallelized features extraction of extract_frequency_bands function

# %% [markdown]
# ## Image generation function

# %%

def augment_EEG(features, std_mult, pca=False, n_components=2):
    # Placeholder for EEG data augmentation logic.
    # Implement augmentation based on PCA or noise addition as required.
    pass

def gen_images(features,locs = AZ_PROJ_VALUE, n_gridpoints = IMG_RESOLUTION, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode.
    """
    n_electrodes = locs.shape[0]  # Number of electrodes
    assert features.shape[1] % n_electrodes == 0, "Features dimension must be divisible by the number of electrodes."
    n_colors = features.shape[1] // n_electrodes # Features.shape = (number_of_epochs, (n_channels*3))
    
    # Reshape features for easier manipulation
    features = features.reshape(-1, n_electrodes, n_colors)# should be checked
    
    # Augment features if required
    if augment:
        features = np.array([augment_EEG(feature_set, std_mult, pca, n_components) for feature_set in features.transpose(2, 0, 1)]).transpose(1, 2, 0)
    
    n_samples = features.shape[0] # number_of_epochs
    grid_x, grid_y = np.mgrid[
        min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
        min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
    ]
    
    # Prepare for edgeless images if needed
    if edgeless:
        edge_locs = np.array([
            [locs[:, 0].min(), locs[:, 1].min()],
            [locs[:, 0].min(), locs[:, 1].max()],
            [locs[:, 0].max(), locs[:, 1].min()],
            [locs[:, 0].max(), locs[:, 1].max()]
        ])
        locs = np.vstack([locs, edge_locs])
        features = np.hstack([features, np.zeros((n_samples, 4, n_colors))])

    # Interpolate features onto the image grid
    interpolated = np.empty((n_samples, n_gridpoints, n_gridpoints, n_colors))
    for i in range(n_samples):
        for c in range(n_colors):
            interpolated[i, :, :, c] = griddata(
                locs, features[i, :, c], (grid_x, grid_y), method='cubic', fill_value=np.nan
            )

    # Normalize if required
    if normalize:
        interpolated = np.where(np.isnan(interpolated), np.nan, scale(interpolated.reshape(-1, n_colors)).reshape(interpolated.shape))

    interpolated = np.nan_to_num(interpolated)
    
    # [samples, W, H, colors]
    return interpolated



# %% [markdown]
# ## Test Generate image

# %% [markdown]
# ## RGB version of the function

# %%
def augment_EEG(features, std_mult, pca=False, n_components=2):
    # Placeholder for EEG data augmentation logic.
    pass

def normalize_and_combine_channels(interpolated):
    """
    Normalize each channel to [0, 1] and combine them into an RGB image.
    Assumes interpolated shape is [samples, width, height, colors].
    """
    n_samples, width, height, _ = interpolated.shape
    rgb_images = np.zeros((n_samples, width, height, 3))  # Initialize RGB image container

    for i in range(n_samples):
        for c in range(3):  # Assuming the first 3 channels correspond to RGB
            min_val = np.nanmin(interpolated[i, :, :, c])
            max_val = np.nanmax(interpolated[i, :, :, c])
            if max_val > min_val:  # Avoid division by zero
                rgb_images[i, :, :, c] = (interpolated[i, :, :, c] - min_val) / (max_val - min_val)
            else:
                rgb_images[i, :, :, c] = np.nan_to_num(interpolated[i, :, :, c])
    return rgb_images

def gen_images_rgb(features,locs = AZ_PROJ_VALUE, n_gridpoints = IMG_RESOLUTION, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode,
    with an added functionality to superpose image channels into an RGB image.
    """
    n_electrodes = locs.shape[0]
    assert features.shape[1] % n_electrodes == 0, "Features dimension must be divisible by the number of electrodes."
    n_colors = features.shape[1] // n_electrodes
    features = features.reshape(-1, n_electrodes, n_colors) # Reshaping here???
    
    if augment:
        features = np.array([augment_EEG(feature_set, std_mult, pca, n_components) for feature_set in features.transpose(2, 0, 1)]).transpose(1, 2, 0)
    
    n_samples = features.shape[0]
    grid_x, grid_y = np.mgrid[
        min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
        min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
    ]
    
    if edgeless:
        locs = np.vstack([locs, [[locs[:, 0].min(), locs[:, 1].min()], [locs[:, 0].min(), locs[:, 1].max()],
                                 [locs[:, 0].max(), locs[:, 1].min()], [locs[:, 0].max(), locs[:, 1].max()]]])
        features = np.hstack([features, np.zeros((n_samples, 4, n_colors))])

    interpolated = np.empty((n_samples, n_gridpoints, n_gridpoints, n_colors))
    for i in range(n_samples):
        for c in range(n_colors):
            interpolated[i, :, :, c] = griddata(
                locs, features[i, :, c], (grid_x, grid_y), method='cubic', fill_value=np.nan
            )

    if normalize:
        rgb_images = normalize_and_combine_channels(interpolated)
    else:
        rgb_images = interpolated[:, :, :, :3]  # Assuming the first 3 channels can directly form an RGB image without normalization

    return rgb_images


# %% [markdown]
# ## Testing extract_frequency_bands

# %%
import numpy as np
import utils  # Assuming this is a custom module with the eeg_to_epochs function
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_frequency_bands_log(file_path, freq_bands=BANDS_FREQ):
    """
    Extracts average power spectral density (PSD) in specified frequency bands for each channel
    in each epoch of an MNE Epochs object.

    Parameters:
    - file_path: raw file path containing the EEG signal.
    - freq_bands: Dictionary with frequency band names as keys and tuples (fmin, fmax) as values.

    Returns:
    - numpy.ndarray of shape [number_of_epochs, (n_channels*3)], where each element represents
      the average PSD in a frequency band for a channel in an epoch.
    """
    logger.info(f"Loading EEG epochs from {file_path}.")
    epochs = utils.eeg_to_epochs(file_path)
    logger.info(f"Loaded {len(epochs)} epochs for processing. Epochs shape: {epochs.get_data().shape}")
    
    results = []

    for idx, epoch in enumerate(epochs):
        logger.debug(f"Processing epoch {idx+1}/{len(epochs)}. Epoch shape: {epoch.shape}")
        
        # Apply Hann window to minimize spectral leakage
        hann_window = np.hanning(epoch.shape[1])
        windowed_epoch = epoch * hann_window
        logger.debug(f"Windowed epoch shape: {windowed_epoch.shape}")
        
        # Perform FFT and calculate power spectral density (PSD)
        fft_epoch = np.fft.rfft(windowed_epoch, axis=1)
        freqs = np.fft.rfftfreq(epoch.shape[1], d=1/epochs.info['sfreq'])
        psd_epoch = np.abs(fft_epoch)
        logger.debug(f"PSD epoch shape: {psd_epoch.shape}, Frequency bins shape: {freqs.shape}")
        
        epoch_features = []
        for i_channel in range(len(epochs.ch_names)):
            for band_name, band in freq_bands.items():
                fmin, fmax = band
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                band_power = psd_epoch[i_channel, freq_mask].mean()
                epoch_features.append(band_power)
        
        results.append(epoch_features)
        logger.info(f"Features extracted for epoch {idx+1}: {len(epoch_features)} features.")

    results_array = np.array(results)
    logger.info(f"Completed extracting frequency bands. Results shape: {results_array.shape}")
    return results_array


# %%
def process_and_save_file_images(path,output_dir=OUTPUT_DIR,bands = BANDS_FREQ):
    features = extract_frequency_bands_log(path)
    images = gen_images(features)  
    image_name = os.path.basename(path).split(".")
    if len(image_name) == 2:
        file_name = f"{image_name[0]}.npy"
    else:
        file_name = f"{image_name[0]+'_'+image_name[1]}.npy"
    np.save(os.path.join(output_dir, file_name), images)


# %%
def process_and_save_file_epochs(path,output_dir=OUTPUT_DIR_EPOCHS):
    epochs = np.array(utils.eeg_to_epochs(path).get_data())  # get the data from the EEG object and convert it to an array of epochs
    epochs_name = utils.extract_filename_without_extension(path)
    np.save(os.path.join(output_dir, epochs_name), epochs)


# %%
process_and_save_file_images(EEG_PATH_FILE[251])

# %% [markdown]
# ## Test save epochs files

# %%
process_and_save_file_epochs(EEG_PATH_FILE[251])

# %%
import time
img_file = 'C:\\Users\\Administrator\\Downloads\\EEG_images\\sch.npy'
img = np.load(img_file)
print(img.shape)
plt.imshow(img[6])
plt.axis('off')


# %%
print(img[6])

# %% [markdown]
# ## Jobs parallelization

# %% [markdown]
# ### Images

# %%
from joblib import Parallel, delayed
launch_again = True # Very heavy, launch again only if neccessary
if launch_again:
    Parallel(n_jobs=-1)(delayed(process_and_save_file_images)(path) for path in EEG_PATH_FILE)

# %%
launch_again = True # Very heavy, launch again only if neccessary
if launch_again:
    Parallel(n_jobs=-1)(delayed(process_and_save_file_epochs)(_path) for _path in EEG_PATH_FILE)


# %% [markdown]
# ## Testing data preprocessing

# %%
pth1 = 'C:\\Users\\Administrator\\Downloads\\EEG_images\\ale.npy'
pth2 = 'C:\\Users\\Administrator\\Downloads\\EEG_images\\kir.npy'
pth3 = 'C:\\Users\\Administrator\\Downloads\\EEG_images\\cha.npy'
pth4 = 'C:\\Users\\Administrator\\Downloads\\EEG_images\\har_1.npy'
im1 = np.load(pth1)
im2 = np.load(pth2)
im3 = np.load(pth3)
im4 = np.load(pth4)
print(f"Image general shape is {im1.shape}")
names = ['ale','kir','cha','har_1']
rgb_images = [im1[6],im2[68],im3[44],im4[12]]
for image in rgb_images:
    print(f"Images in the file shape : {image.shape}")

# %%
# Create a figure to hold the subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Flatten the array of axes for easy iteration
axs_flat = axs.flatten()

# Loop through the first 4 images and their corresponding axes
for i, ax in enumerate(axs_flat):
    # Select the ith image
    img = rgb_images[i]
    
    # Display the image on the ith subplot
    ax.imshow(img)
    ax.set_title(f'EEG {names[i]}')
    ax.axis('off')  # Hide axes for better visualization

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()


# %% [markdown]
# ## Futher data processing

# %%

def stack_arrays_vertically_memmap(file_paths, output_file,burnout_to_name = df_filtered ):
    """
    Load large NumPy arrays from file paths as memory-mapped files and
    stack them vertically in a memory-efficient way.

    Parameters:
    - file_paths: List of strings, paths to the NumPy array files.
    - output_file: String, path to the output file for the stacked array.

    Returns:
    - A memory-mapped NumPy array of the stacked arrays.
    """
    # Memory-map the first array to determine the shape and dtype
    first_array = np.load(file_paths[0], mmap_mode='r')
    array_shape = first_array.shape
    array_dtype = first_array.dtype
    
    # Calculate the total shape of the stacked array
    total_rows = sum(np.load(path, mmap_mode='r').shape[0] for path in file_paths)
    stacked_shape = (total_rows,) + array_shape[1:]
    
    # Create a memory-mapped array with the total shape
    stacked_array = np.memmap(output_file, dtype=array_dtype, mode='w+', shape=stacked_shape)
    stacked_burnout = np.zeros((total_rows,))  # To keep track of burnout variable
    # Fill the memory-mapped array with data from the original arrays
    current_position = 0
    for path in file_paths:
        _name = str(utils.extract_filename_without_extension(path))
        memmap_array = np.load(path, mmap_mode='r')
        stacked_burnout[current_position:current_position + memmap_array.shape[0]] = burnout_to_name[burnout_to_name['name']==_name]['burnout_Boiko']
        stacked_array[current_position:current_position + memmap_array.shape[0], ...] = memmap_array
        current_position += memmap_array.shape[0]
    
    return stacked_array,stacked_burnout



# %% [markdown]
# ## Processed images and epochs saved

# %%
import glob

path_file_images = glob.glob('C:\\Users\\Administrator\\Downloads\\EEG_images\\*.npy')
path_file_epochs =  glob.glob('C:\\Users\\Administrator\\Downloads\\EEG_epochs\\*.npy')
print(len(path_file_images))  # Number of files in the folder
print(len(path_file_epochs))   # Number of epochs, should be equal to number of images
assert len(path_file_images) == len(path_file_epochs), "Number of EEG image and Epoch file is not match"

# %% [markdown]
# ### Preprocessed images 

# %%
if True:
    output_file = 'preprocessed_images.npy'
    stacked_array,test_file = stack_arrays_vertically_memmap(path_file_images,output_file)
    # Accessing the stacked_array will read from the disk as needed
    print(stacked_array.shape) 

# %%
print(stacked_array.shape)

# %% [markdown]
# ### Preprocessed epochs

# %%
if True:
    output_file = 'preprocessed_epochs.npy'
    stacked_array_epochs,_ = stack_arrays_vertically_memmap(path_file_epochs,output_file)

    # Accessing the stacked_array will read from the disk as needed
    print(stacked_array_epochs.shape)

# %%
print(stacked_array_epochs.shape)

# %% [markdown]
# ## Save labels

# %%
output_dir = 'C:\\Users\\Administrator\\Downloads\\EEG_preprocessed_data'
file_name = 'burnout_level.npy'
output_file_path = os.path.join(output_dir, file_name)

# Check if the directory exists, if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Use numpy.save to save the array to the specified file
np.save(output_file_path, test_file)
np.save(output_dir+'\\preprocessed_epochs.npy', stacked_array_epochs)
np.save(output_dir+'\\preprocessed_images.npy', stacked_array)
np.save(output_dir+file_name, test_file)

# %%
test_file
np.save(output_dir+file_name, test_file)

# %%
print(sum(test_file>61.0)/test_file.shape[0])

# %%
output_dir = 'C:\\Users\\Administrator\\Downloads\\EEG_preprocessed_data'
file_name = 'burnout_level.npy'
output_file_path = os.path.join(output_dir, file_name)

# Check if the directory exists, if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Use numpy.save to save the array to the specified file
np.save(output_file_path, stacked_array_epochs)


