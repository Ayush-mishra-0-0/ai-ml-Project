import os
from pathlib import Path
from PIL import Image
import concurrent.futures
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def convert_image(file_path):
    """Convert a single image from raw.jpg to jpg format."""
    try:
        # Open the image
        img = Image.open(file_path)
        
        # Create new filename by removing 'raw.' from the name
        new_name = str(file_path).replace('raw.jpg', 'jpg')
        
        # Save as regular JPEG
        img.save(new_name, 'JPEG', quality=95)
        
        # Close the image to free up memory
        img.close()
        
        # Remove original file only if new file exists
        if os.path.exists(new_name):
            os.remove(file_path)
            
        return True, file_path
    except Exception as e:
        return False, f"Error converting {file_path}: {str(e)}"

def convert_directory(input_dir):
    """Convert all raw.jpg images in a directory to jpg format."""
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Directory not found: {input_dir}")
    
    # Find all raw.jpg files
    raw_files = list(input_dir.glob('**/*raw.jpg'))
    total_files = len(raw_files)
    
    if total_files == 0:
        logging.info("No raw.jpg files found!")
        return
    
    logging.info(f"Found {total_files} raw.jpg files to convert")
    
    # Convert files using multiple threads
    success_count = 0
    error_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Use tqdm for progress bar
        futures = [executor.submit(convert_image, f) for f in raw_files]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Converting images"):
            success, result = future.result()
            if success:
                success_count += 1
            else:
                error_count += 1
                logging.error(result)
    
    logging.info(f"""
Conversion completed:
- Successfully converted: {success_count} files
- Errors: {error_count} files
""")

if __name__ == "__main__":
    # Replace this with your image directory path
    image_dir = "dataset/data_images/Extracted Images"
    
    try:
        convert_directory(image_dir)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise