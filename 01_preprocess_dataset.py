import datasets
import re
import os
import yaml

# Load configuration from YAML file
def load_config(yaml_file="config.yaml"):
    """
    Load the configuration settings from a YAML file.

    Args:
        yaml_file (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Initialize counters
non_ascii_count = 0
extra_space_count = 0
total_chars_before = 0
total_chars_after = 0

# Basic text preprocessing function with counting
def preprocess_text(text, config):
    """
    Clean and normalize text by:
    - Removing non-ASCII characters (optional)
    - Lowercasing all characters (optional)
    - Removing extra spaces (optional)
    
    Counts the number of removed non-ASCII characters and extra spaces.

    Args:
        text (str): The raw input text.
        config (dict): Preprocessing configuration settings.

    Returns:
        str: Cleaned and normalized text.
    """
    global non_ascii_count, extra_space_count, total_chars_before, total_chars_after
    
    # Count total characters before preprocessing
    total_chars_before += len(text)

    # Remove non-ASCII characters if enabled in config
    if config['preprocessing']['remove_non_ascii']:
        non_ascii_count += len(re.findall(r'[^\x00-\x7F]+', text))
        text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Lowercase if enabled in config
    if config['preprocessing']['lowercase']:
        text = text.lower()
    
    # Remove extra spaces if enabled in config
    if config['preprocessing']['remove_extra_spaces']:
        initial_length = len(text)
        text = re.sub(r'\s+', ' ', text).strip()
        extra_space_count += initial_length - len(text)

    # Count total characters after preprocessing
    total_chars_after += len(text)
    
    return text

# Preprocess the dataset by applying the cleaning function to each text sample
def preprocess_dataset(dataset, config):
    """
    Apply preprocessing to the entire dataset (train, validation, test),
    while also keeping track of removed characters and spaces.

    Args:
        dataset: The Hugging Face dataset object.
        config: The configuration dictionary.

    Returns:
        dataset: The preprocessed dataset.
    """
    for split in dataset.keys():
        dataset[split] = dataset[split].map(lambda example: {"text": preprocess_text(example['text'], config)})
    return dataset

# Main function to load, preprocess, and save the dataset
def main():
    # Load the configuration
    config = load_config()

    # Load the dataset based on the config
    dataset_name = config['dataset']['name']
    dataset_version = config['dataset']['version']
    dataset = datasets.load_dataset(dataset_name, dataset_version)

    # Preprocess the dataset
    preprocessed_dataset = preprocess_dataset(dataset, config)

    # Specify the path to save the preprocessed dataset
    dataset_save_path = config['dataset']['dataset_save_path']

    # Create the directory if it doesn't exist
    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path)

    # Save the preprocessed dataset to disk
    preprocessed_dataset.save_to_disk(dataset_save_path)

    # Calculate percentages
    removed_non_ascii_percentage = (non_ascii_count / total_chars_before) * 100 if total_chars_before > 0 else 0
    removed_spaces_percentage = (extra_space_count / total_chars_before) * 100 if total_chars_before > 0 else 0

    # Print the statistics
    print(f"Preprocessed dataset saved to {dataset_save_path}")
    print(f"Total non-ASCII characters removed: {non_ascii_count}")
    print(f"Total unwanted spaces removed: {extra_space_count}")
    print(f"Removed non-ASCII characters as a percentage of total characters: {removed_non_ascii_percentage:.2f}%")
    print(f"Removed unwanted spaces as a percentage of total characters: {removed_spaces_percentage:.2f}%")

# Run the main function
if __name__ == "__main__":
    main()
