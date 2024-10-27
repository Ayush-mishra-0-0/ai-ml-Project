import pandas as pd
import os
from tqdm import tqdm

def filter_csv_by_images(csv_path, image_folder, image_path_column):
    """
    Creates a new CSV containing only rows where corresponding images exist.
    
    Parameters:
    -----------
    csv_path : str
        Path to the original CSV file
    image_folder : str
        Path to the folder containing images
    image_path_column : str
        Name of the column in CSV containing image paths/names
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame containing only rows with existing images
    """
    # Read the original CSV
    print("Reading CSV file...")
    df = pd.read_csv(csv_path)
    original_count = len(df)
    
    # Function to check if image exists
    def image_exists(row):
        img_path = os.path.join(image_folder, row[image_path_column])
        return os.path.exists(img_path)
    
    # Filter rows using tqdm for progress tracking
    print("Checking image existence...")
    tqdm.pandas()
    mask = df.progress_apply(image_exists, axis=1)
    filtered_df = df[mask].copy()
    
    # Print statistics
    remaining_count = len(filtered_df)
    removed_count = original_count - remaining_count
    
    print("\nFiltering Statistics:")
    print(f"Original rows: {original_count:,}")
    print(f"Remaining rows: {remaining_count:,}")
    print(f"Removed rows: {removed_count:,}")
    print(f"Retention rate: {(remaining_count/original_count)*100:.2f}%")
    
    # Save filtered CSV
    output_path = csv_path.rsplit('.', 1)[0] + '_filtered.csv'
    filtered_df.to_csv(output_path, index=False)
    print(f"\nFiltered CSV saved to: {output_path}")
    
    return filtered_df

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual paths
    csv_path = r"dataset\cloud_data_cleaned1.csv"
    image_folder = r"dataset\data_images\Extracted Images"
    image_path_column = "image_name"  # Replace with your column name
    
    filtered_df = filter_csv_by_images(csv_path, image_folder, image_path_column)