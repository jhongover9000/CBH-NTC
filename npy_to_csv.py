import numpy as np
import pandas as pd

# Load the .npy file
npy_file = 'generated_labels_balanced_v3.npy'  # Replace with your .npy file path
data = np.load(npy_file)

# Convert the numpy array to a pandas DataFrame for easy conversion to CSV
df = pd.DataFrame(data)

from collections import Counter

# Convert to CSV
csv_file = 'output_file.csv'  # Output file path for CSV
df.to_csv(csv_file, index=False)  # Save DataFrame as CSV without index


# Flatten the data to ensure it's one-dimensional for counting
data_flat = data.flatten()

# Use Counter to count the occurrences of each item
item_counts = Counter(data_flat)

# Display the counts
for item, count in item_counts.items():
    print(f"Item: {item}, Count: {count}")


# Alternatively, if you want to save as TXT, you can use the following:
# txt_file = 'output_file.txt'  # Output file path for TXT
# np.savetxt(txt_file, data, delimiter=',')  # Save as .txt with comma delimiter
