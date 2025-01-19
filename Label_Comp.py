import numpy as np

# Load previous and new labels
rule_based_labels = np.load("generated_labels_balanced_v1.npy")  # Labels from the rule-based method
kmeans_labels = np.load("generated_labels_ind_diff.npy")  # Labels from K-means clustering

# Ensure labels are the same length
assert rule_based_labels.shape == kmeans_labels.shape, "Label arrays must have the same shape!"

# Find trials where both methods assigned labels (exclude -1s)
valid_mask = rule_based_labels != -1  # Ignore trials where rule-based method failed

# Compute match index
matching_labels = np.sum(rule_based_labels[valid_mask] == kmeans_labels[valid_mask])
total_valid = np.sum(valid_mask)
match_index = (matching_labels / total_valid) * 100 if total_valid > 0 else 0

# Print results
print(f"Total Trials: {len(rule_based_labels)}")
print(f"Labeled Trials: {total_valid}")
print(f"Matching Labels: {matching_labels}")
print(f"Match Index: {match_index:.2f}%")

# Save comparison results
np.save("label_comparison.npy", {"match_index": match_index, 
                                 "rule_based_labels": rule_based_labels, 
                                 "kmeans_labels": kmeans_labels})
