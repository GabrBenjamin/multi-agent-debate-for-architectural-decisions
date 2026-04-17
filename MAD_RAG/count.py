import pandas as pd

# Extract the values from the column "comparison_result" and filter for "No", "Yes", and "Maybe"
data = pd.read_csv(r'output_with_comparisonsATAM_GLGL_ALL.csv')
def count_yes_no_maybe(data, column_name):
    # Initialize lists to store the strings
    no_list = []
    yes_list = []
    maybe_list = []

    # Loop through the column to categorize the entries
    for entry in data[column_name]:
        # Extract the main string (e.g., "No", "Yes", "Maybe") before any additional text
        cleaned_entry = entry.split('//')[0].strip()

        if "No" in cleaned_entry:
            no_list.append(cleaned_entry)
        elif "Yes" in cleaned_entry:
            yes_list.append(cleaned_entry)
        elif "Maybe" in cleaned_entry:
            maybe_list.append(cleaned_entry)

    # Return counts and the lists
    return len(no_list), len(yes_list), len(maybe_list), no_list, yes_list, maybe_list

# Example usage
# Assuming `data` is a pandas DataFrame containing the "comparison_result" column
no_count, yes_count, maybe_count, no_list, yes_list, maybe_list = count_yes_no_maybe(data, "comparison_result")
print(f"No Count: {no_count}, Yes Count: {yes_count}, Maybe Count: {maybe_count}")

def count_negatives(data, comparison_column, supported_side_column):
    # Extract all rows where supported_side is "Negative"
    negative_ids = data[data[supported_side_column] == "Negative"].index.tolist()

    # Get counts and lists from the comparison_column
    no_count, yes_count, maybe_count, no_list, yes_list, maybe_list = count_yes_no_maybe(data, comparison_column)

    # Initialize counters for matches with "Negative"
    no_negative_count = len([idx for idx in data[data[comparison_column].str.contains("No")].index if idx in negative_ids])
    yes_negative_count = len([idx for idx in data[data[comparison_column].str.contains("Yes")].index if idx in negative_ids])
    maybe_negative_count = len([idx for idx in data[data[comparison_column].str.contains("Maybe")].index if idx in negative_ids])

    # Return counts and match results
    return no_negative_count, yes_negative_count, maybe_negative_count, negative_ids

# Example usage
# Assuming `data` is a pandas DataFrame containing the "comparison_result" and "supported_side" columns
no_negative_count, yes_negative_count, maybe_negative_count, negative_ids = count_negatives(data, "comparison_result", "supported_side")

print(f"No matching Negative: {no_negative_count}, Yes matching Negative: {yes_negative_count}, Maybe matching Negative: {maybe_negative_count}")

def count_affirmatives(data, comparison_column, supported_side_column):
    # Extract all rows where supported_side is "Affirmative"
    affirmative_ids = data[data[supported_side_column] == "Affirmative"].index.tolist()

    # Get counts and lists from the comparison_column
    no_count, yes_count, maybe_count, no_list, yes_list, maybe_list = count_yes_no_maybe(data, comparison_column)

    # Initialize counters for matches with "Affirmative"
    no_affirmative_count = len([idx for idx in data[data[comparison_column].str.contains("No")].index if idx in affirmative_ids])
    yes_affirmative_count = len([idx for idx in data[data[comparison_column].str.contains("Yes")].index if idx in affirmative_ids])
    maybe_affirmative_count = len([idx for idx in data[data[comparison_column].str.contains("Maybe")].index if idx in affirmative_ids])

    # Return counts and match results
    return no_affirmative_count, yes_affirmative_count, maybe_affirmative_count, affirmative_ids

# Example usage
# Assuming `data` is a pandas DataFrame containing the "comparison_result" and "supported_side" columns
no_affirmative_count, yes_affirmative_count, maybe_affirmative_count, affirmative_ids = count_affirmatives(data, "comparison_result", "supported_side")

print(f"No matching Affirmative: {no_affirmative_count}, Yes matching Affirmative: {yes_affirmative_count}, Maybe matching Affirmative: {maybe_affirmative_count}")
