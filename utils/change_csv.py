import pandas as pd
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_dataset(csv_path, image_dir):
    """
    Clean the dataset by:
    1. Updating CSV file to remove 'raw.' from image names
    2. Renaming actual image files to match
    """
    # Load CSV file
    logging.info(f"Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} rows")
    
    # Create backup of original CSV
    csv_backup = csv_path.replace('.csv', '_backup.csv')
    df.to_csv(csv_backup, index=False)
    logging.info(f"Created CSV backup at: {csv_backup}")
    
    # Update image names in DataFrame
    logging.info("Updating image names in CSV...")
    df['image_name'] = df['image_name'].str.replace('raw.jpg', 'jpg')
    
    # Create mapping of old to new names
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Image directory not found: {image_dir}")
    
    # Create backup directory for original images
    backup_dir = image_dir.parent / "Original_Images_Backup"
    backup_dir.mkdir(exist_ok=True)
    logging.info(f"Created backup directory at: {backup_dir}")
    
    # Rename image files
    logging.info("Processing image files...")
    renamed_count = 0
    errors = []
    
    for old_name in tqdm(df['image_name']):
        old_path = image_dir / old_name.replace('jpg', 'raw.jpg')
        new_path = image_dir / old_name
        backup_path = backup_dir / old_path.name
        
        try:
            if old_path.exists():
                # Backup original file
                shutil.copy2(old_path, backup_path)
                
                # Rename file
                old_path.rename(new_path)
                renamed_count += 1
            else:
                errors.append(f"File not found: {old_path}")
        except Exception as e:
            errors.append(f"Error processing {old_path}: {str(e)}")
    
    # Save updated CSV
    logging.info("Saving updated CSV...")
    df.to_csv(csv_path, index=False)
    
    # Print summary
    logging.info(f"""
Cleaning Complete:
- Created CSV backup at: {csv_backup}
- Created image backup at: {backup_dir}
- Successfully renamed {renamed_count} files
- Encountered {len(errors)} errors
""")
    
    if errors:
        logging.info("\nFirst 10 errors encountered:")
        for error in errors[:10]:
            logging.error(error)

if __name__ == "__main__":
    # Update these paths to match your setup
    csv_path = "dataset/cloud_data_cleaned1.csv"
    image_dir = "dataset/data_images/Extracted Images"
    
    try:
        clean_dataset(csv_path, image_dir)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise