# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import soundfile as sf

# Load the data.tsv file
#data = pd.read_csv('filtered_data_labeled.tsv', sep='\t')

# %%
# plot count of each unique label for each column (up_votes	down_votes	age	gender	accent	label)
# up_votes divided by down_votes
# Handle infinite values caused by division by zero
data['up_votes-down_votes'] = data['up_votes'] - data['down_votes']
data['up_votes-down_votes'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values in 'up_votes-down_votes' or handle them as needed
data['up_votes-down_votes'].dropna(inplace=True)

# Plot the histogram
counts, bins = np.histogram(data['up_votes-down_votes'].dropna(), bins=255)

# Print the counts in each bin
for i in range(len(bins) - 1):
    print(f"Bin range: {bins[i]:.2f} - {bins[i+1]:.2f}, Count: {counts[i]}")

# Plot the histogram
data['up_votes-down_votes'].plot(kind='hist', bins=100, title='up_votes-down_votes')


# %%
# get worst rated row
worst_rated_row = data.loc[data['up_votes-down_votes'].idxmin()]
print(worst_rated_row)

# %%
# drop all columns except path and label
data = data[['path', 'label']]
# remove duplicate paths rows
data = data.drop_duplicates(subset=['path'])
# remove rows with empty paths
data = data[data['path'].notna()]
# remove rows with empty labels
data = data[data['label'].notna()]


# %%
## plot the classes distribution
# get the unique labels and their counts
labels = data['label'].value_counts()
# plot the labels distribution
labels.plot(kind='bar', title='Labels Distribution')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# Clearly, the data is biased (class imbalance), around 70% of the examples belong to class 0.
# To solve this problem we have to balance the dataset. We can do this by oversampling the minority class or undersampling the majority class.

# %%
# get the names of the actual files under data directory to remove paths that do not exist
import os
files = os.listdir('data')
actual_files_count = len(files)
data_files_count = len(data['path'])
print(f"Actual files count: {actual_files_count}")
print(f"Data files count: {data_files_count}")
# remove paths that do not exist
data = data[data['path'].isin(files)]


# %%
actual_files_count = len(files)
data_files_count = len(data['path'])
print(f"Actual files count: {actual_files_count}")
print(f"Data files count: {data_files_count}")

# %%
#write the data to a new csv file
data.to_csv('filtered_data_labeled_cleaned.csv', index=False)

# %%
# plot sample rates for all files (histogram of sample rates)
import soundfile as sf
import os


sample_rates = []
with open('corrupt_files.txt', 'w') as f:
	for path in data['path']:
		try:
			# get the sample rate of the file
			sr = sf.read(os.path.join('data', path))[1]
			sample_rates.append(sr)
		except Exception as e:
			print(f"Error reading {path}: {e}")
			f.write(path + '\n')
			data = data[data['path'] != path]

# plot the histogram of sample rates
plt.hist(sample_rates, bins=100, color='blue', alpha=0.7)
plt.title('Sample Rates Distribution')
plt.xlabel('Sample Rate')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# %%
#get unique sample rates and their counts
unique_sample_rates = pd.Series(sample_rates).value_counts()
print(unique_sample_rates)

# %% [markdown]
# same sample rate for all audio files

# %%
data.to_csv('filtered_data_labeled_cleaned_working.csv', index=False)

# %% [markdown]
# ## Selection

# %%
# load data
data = pd.read_csv('filtered_data_labeled_cleaned_working.csv')

# %% [markdown]
# we want to select a subset of the dataset that is representative of the whole dataset. We can do this by using stratified sampling.

# %%
data = data.sample(n=10000)
file_sizes = []
progress = 0
for path in data['path']:
    try:
        # get the size of the file in bytes
        file_size = os.path.getsize(os.path.join('data', path))
        file_sizes.append(file_size)
        progress += 1
        if progress % 100 == 0:
            print(f"Processed {progress} files")
    except Exception as e:
        print(f"Error reading {path}: {e}")
        data = data[data['path'] != path]
# plot the histogram of file sizes
plt.hist(file_sizes, bins=100, color='blue', alpha=0.7)
plt.title('File Sizes Distribution')
plt.xlabel('File Size (bytes)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
# get the mean file size
mean_file_size = np.mean(file_sizes)
print(f"Mean file size: {mean_file_size} bytes")
# get the std file size
std_file_size = np.std(file_sizes)
print(f"Std file size: {std_file_size} bytes")

# %%
# get count of files less than 5KB
less_than_5KB = len([f for f in file_sizes if f < 5 * 1024])
print(f"Count of files less than 5KB: {less_than_5KB}")
# get min file size
min_file_size = np.min(file_sizes)
print(f"Min file size: {min_file_size} bytes")

# %%
def is_file_size_in_range(file_path, min_size=25 * 1024, max_size=50 * 1024):
    try:
        # Check if the file exists and is not empty
        return os.path.exists(file_path) and os.path.getsize(file_path) > min_size and os.path.getsize(file_path) < max_size
    except Exception as e:
        print(f"Error checking file {file_path}: {e}")
        return False

# %%
# get 10000 random samples from each class, but make sure to check if the file size is in range is_file_size_in_range function
def get_random_samples(data, n=10000):
    # get the unique labels and their counts
    labels = data['label'].value_counts()
    # get the unique labels
    unique_labels = labels.index.tolist()
    # create a new dataframe to store the samples
    samples = pd.DataFrame(columns=data.columns)
    for label in unique_labels:
        # get the paths of the files with the label
        paths = data[data['label'] == label]['path'].tolist()
        # shuffle the paths
        np.random.shuffle(paths)
        # get the first n paths that are in range
        paths_in_range = [p for p in paths if is_file_size_in_range(os.path.join('data', p))]
        # get the first n paths
        paths_in_range = paths_in_range[:n]
        # add the samples to the dataframe
        samples = pd.concat([samples, data[data['path'].isin(paths_in_range)]])
    # shuffle the samples randomly
    samples = samples.sample(frac=1).reset_index(drop=True)
    # save the samples to a new csv file
    samples.to_csv('filtered_data_labeled_cleaned_working_samples.csv', index=False)

# get 10000 random samples from each class
get_random_samples(data, n=10000)


# %%
data = data.sample(n=10000)
durations = []
progress = 0
for path in data['path']:
    try:
        # get the duration of the file
        duration = sf.read(os.path.join('data', path))[0].shape[0] / sf.read(os.path.join('data', path))[1]
        durations.append(duration)
        progress += 1
        if progress % 100 == 0:
            print(f"Processed {progress} files")
    except Exception as e:
        print(f"Error reading {path}: {e}")
        data = data[data['path'] != path]
# plot the histogram of durations
plt.hist(durations, bins=100, color='blue', alpha=0.7)
plt.title('Durations Distribution')
plt.xlabel('Duration')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
# get the mean duration
mean_duration = np.mean(durations)
print(f"Mean duration: {mean_duration}")
# get the std duration
std_duration = np.std(durations)
print(f"Std duration: {std_duration}")

# %% [markdown]
# ## Denoising

# %%
# from multiprocessing import Pool, cpu_count
# import os
# import librosa
# import soundfile as sf

# def denoise_file(file_path):
#     try:
#         # Construct full input path
#         input_path = os.path.join('data', file_path)
        
#         # Read with explicit error handling
#         y, sr = librosa.load(input_path, sr=None, mono=True)
        
#         # Apply preemphasis (note: this is NOT a low-pass filter)
#         y_denoised = librosa.effects.preemphasis(y, coef=0.97)
        
#         # Ensure output directory exists
#         os.makedirs('denoised', exist_ok=True)
        
#         # Save with original sample rate
#         output_path = os.path.join('denoised', os.path.basename(file_path))
#         sf.write(output_path, y_denoised, sr)
#         return f"Success: {file_path}"
#     except Exception as e:
#         return f"Failed {file_path}: {str(e)}"

# if __name__ == '__main__':
#     # Get file paths
#     file_paths = data['path'].tolist()[0:40]  # First 40 files
    
#     # Create pool with logical CPUs
#     with Pool(processes=cpu_count()) as pool:
#         results = pool.map(denoise_file, file_paths)
    
#     # Print results
#     for result in results:
#         print(result)

# %%
#normalize the format and the sample rate of the audio files
import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import shutil
import random
import glob
import re

# Define the directory containing the audio files
audio_dir = 'data'
# Define the directory to save the normalized audio files
output_dir = 'normalized_data'
# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Define the target sample rate (get minimum sample rate from the data)


