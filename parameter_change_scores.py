import pandas as pd

def calculate_averages(file_path, parameters):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Check if 'parameters' is a string or list and filter accordingly
    if isinstance(parameters, str):
        filtered_df = df[df['Key'].str.contains(parameters)]
    elif isinstance(parameters, list):
        filtered_df = df
        for param in parameters:
            filtered_df = filtered_df[filtered_df['Key'].str.contains(param)]
    else:
        raise ValueError("Parameters must be a string or a list of strings.")
    
    # Calculate the averages for each column, excluding the 'Combination' column
    if not filtered_df.empty:
        averages = filtered_df.iloc[:, 1:].mean().to_dict()
    else:
        averages = {col: 0 for col in df.columns[1:]}
    
    return averages


def calculate_multiple_averages(file_path, combinations, output_file):
    results = []
    for combination in combinations:
        averages = calculate_averages(file_path, combination)
        results.append({
            "Parameters": str(combination),
            **averages
        })
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_file, index=False)
    return results


if __name__ == "__main__":
    file_path = "output-finite/Res_m2.xlsx"
    output_path = "output-finite/Res_m2_variations.xlsx"
    
    # Define the combinations to evaluate
    combinations = [
        "ut1",
        ["ut2", "uo4"],
        ["ut2", "uo8"],
        ["ut2", "uo16"],
        ["ut3", "uo4"],
        ["ut3", "uo8"],
        ["ut3", "uo16"],
        "th0.1", "th0.2", "th0.3", "th0.4", "th0.5", "th0.6", "th0.7", "th0.8", "th0.9"
    ]
    
    # Calculate the averages
    calculate_multiple_averages(file_path, combinations, output_path)
    
