# Function to calculate the probability that the original F1 score is greater than the average of the two groups after splitting

from sklearn.metrics import f1_score
import numpy as np

# Initial values for confusion matrix
TP = 50
FP = 10
FN = 10
TN = 30

# Calculating F1 score for the initial confusion matrix
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_initial = 2 * (precision * recall) / (precision + recall)


# Function to simulate splitting the data into two groups and calculating F1 score for each
def simulate_split_f1(TP, FP, FN, TN, split_even=True):
    """
    Simulates splitting the data into two groups and calculates the F1 score for each group.

    :param TP: int - True Positives
    :param FP: int - False Positives
    :param FN: int - False Negatives
    :param TN: int - True Negatives
    :param split_even: bool - If True, splits the data into even groups; if False, splits randomly
    :return: tuple - (f1_group1, f1_group2, f1_initial)
    """
    # Total number of samples
    total_samples = TP + FP + FN + TN
    # Simulate splitting into two groups
    if split_even:
        group_samples = total_samples // 2
    else:
        group_samples = np.random.randint(1, total_samples)

    # Randomly assign each sample to a group
    group1 = np.random.choice(['TP', 'FP', 'FN', 'TN'], group_samples,
                              p=[TP / total_samples, FP / total_samples, FN / total_samples, TN / total_samples])
    group2 = np.random.choice(['TP', 'FP', 'FN', 'TN'], total_samples - group_samples,
                              p=[TP / total_samples, FP / total_samples, FN / total_samples, TN / total_samples])

    # Count the occurrences of each type for both groups
    TP1, FP1, FN1, TN1 = np.bincount([{'TP': 0, 'FP': 1, 'FN': 2, 'TN': 3}[x] for x in group1], minlength=4)
    TP2, FP2, FN2, TN2 = np.bincount([{'TP': 0, 'FP': 1, 'FN': 2, 'TN': 3}[x] for x in group2], minlength=4)

    # Calculate F1 score for each group
    precision1 = TP1 / (TP1 + FP1) if (TP1 + FP1) > 0 else 0
    recall1 = TP1 / (TP1 + FN1) if (TP1 + FN1) > 0 else 0
    precision2 = TP2 / (TP2 + FP2) if (TP2 + FP2) > 0 else 0
    recall2 = TP2 / (TP2 + FN2) if (TP2 + FN2) > 0 else 0
    f1_group1 = 2 * (precision1 * recall1) / (precision1 + recall1) if (precision1 + recall1) > 0 else 0
    f1_group2 = 2 * (precision2 * recall2) / (precision2 + recall2) if (precision2 + recall2) > 0 else 0

    return f1_group1, f1_group2, f1_initial


# Perform multiple simulations and check if both groups have higher F1 score than initial
iterations = 10000
higher_f1_cases_even = []
higher_f1_cases_uneven = []

for _ in range(iterations):
    f1_group1_even, f1_group2_even, _ = simulate_split_f1(TP, FP, FN, TN, split_even=True)
    f1_group1_uneven, f1_group2_uneven, _ = simulate_split_f1(TP, FP, FN, TN, split_even=False)

    # Check if both groups have a higher F1 score than the initial F1 score
    if f1_group1_even > f1_initial and f1_group2_even > f1_initial:
        higher_f1_cases_even.append((f1_group1_even, f1_group2_even))

    if f1_group1_uneven > f1_initial and f1_group2_uneven > f1_initial:
        higher_f1_cases_uneven.append((f1_group1_uneven, f1_group2_uneven))

f1_initial, higher_f1_cases_even, higher_f1_cases_uneven


def calculate_probability_of_f1_comparison(higher_f1_cases):
    """
    Calculates the probability that the original F1 score is greater than or less than the average of two groups' F1 scores.

    :param higher_f1_cases: list - List of tuples containing F1 scores for two groups after splitting
    :return: tuple - (probability_greater, probability_less)
    """
    count_greater = 0
    count_less = 0
    for f1_group1, f1_group2 in higher_f1_cases:
        average_f1_groups = (f1_group1 + f1_group2) / 2
        if f1_initial > average_f1_groups:
            count_greater += 1
        else:
            count_less += 1

    total_cases = len(higher_f1_cases)
    probability_greater = count_greater / total_cases
    probability_less = count_less / total_cases
    return probability_greater, probability_less


# Calculate probabilities for even split
prob_greater_even, prob_less_even = calculate_probability_of_f1_comparison(higher_f1_cases_even)
# Calculate probabilities for uneven split
prob_greater_uneven, prob_less_uneven = calculate_probability_of_f1_comparison(higher_f1_cases_uneven)

print(prob_greater_even, prob_less_even, prob_greater_uneven, prob_less_uneven
)