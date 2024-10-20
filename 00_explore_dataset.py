import datasets
import os

# Disable the caching warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# Load WikiText-2 dataset from Hugging Face
dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')

# Explore the dataset structure
print("Dataset Structure:")
print(dataset)

# Inspect one example from the training set
print("\nSample from Training Set:")
print(dataset['train'][0]['text'])

# Get some statistics on the dataset
train_text = dataset['train']['text']
valid_text = dataset['validation']['text']
test_text = dataset['test']['text']

def dataset_statistics(text_data):
    """
    Calculate and print basic statistics of the dataset:
    - Number of samples
    - Total number of characters
    - Average characters per sample

    Args:
        text_data (List[str]): A list of text samples.
    """
    num_samples = len(text_data)
    total_chars = sum(len(text) for text in text_data)
    avg_chars_per_sample = total_chars / num_samples

    print(f"Number of samples: {num_samples}")
    print(f"Total number of characters: {total_chars}")
    print(f"Average characters per sample: {avg_chars_per_sample:.2f}")

# Training set statistics
print("\nTraining Set Statistics:")
dataset_statistics(train_text)

# Validation set statistics
print("\nValidation Set Statistics:")
dataset_statistics(valid_text)

# Test set statistics
print("\nTest Set Statistics:")
dataset_statistics(test_text)

# Display a few more sample texts from the dataset
print("\nFirst 5 Samples from the Training Set:")
for i in range(5):
    print(f"Sample {i + 1}: {train_text[i]}")