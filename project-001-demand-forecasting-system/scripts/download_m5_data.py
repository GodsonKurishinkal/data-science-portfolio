"""
Script to download M5 Competition dataset from Kaggle.

This script downloads the M5 Forecasting Accuracy dataset from Kaggle
and extracts it to the data/raw directory.
"""

import os
import zipfile
import subprocess
from pathlib import Path


def download_m5_dataset():
    """
    Download M5 dataset from Kaggle using Kaggle API.
    
    Prerequisites:
    - Kaggle API installed: pip install kaggle
    - Kaggle API credentials configured (~/.kaggle/kaggle.json)
    """
    # Define paths
    project_root = Path(__file__).parent.parent
    data_raw_path = project_root / "data" / "raw"
    
    # Create directory if it doesn't exist
    data_raw_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("M5 Dataset Download")
    print("=" * 60)
    
    # Check if data already exists
    expected_files = [
        "calendar.csv",
        "sales_train_validation.csv",
        "sell_prices.csv"
    ]
    
    existing_files = [f for f in expected_files if (data_raw_path / f).exists()]
    
    if len(existing_files) == len(expected_files):
        print("\n✓ All M5 dataset files already exist in data/raw/")
        print("Files found:")
        for f in existing_files:
            file_size = (data_raw_path / f).stat().st_size / (1024 * 1024)
            print(f"  - {f} ({file_size:.2f} MB)")
        return True
    
    # Check if Kaggle API is installed
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True
        )
        print(f"\n✓ Kaggle API found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("\n✗ Kaggle API not found!")
        print("\nPlease install it using:")
        print("  pip install kaggle")
        print("\nThen configure your API credentials:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New API Token'")
        print("  3. Place kaggle.json in ~/.kaggle/")
        print("  4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Download dataset
    print("\n⏳ Downloading M5 dataset from Kaggle...")
    print("This may take several minutes depending on your connection...")
    
    try:
        subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", "m5-forecasting-accuracy",
                "-p", str(data_raw_path)
            ],
            check=True
        )
        print("✓ Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Download failed: {e}")
        print("\nPlease ensure:")
        print("  1. Kaggle API credentials are configured")
        print("  2. You have accepted the competition rules at:")
        print("     https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules")
        return False
    
    # Extract zip file
    zip_path = data_raw_path / "m5-forecasting-accuracy.zip"
    
    if zip_path.exists():
        print(f"\n⏳ Extracting files...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_raw_path)
            print("✓ Extraction complete!")
            
            # Remove zip file
            zip_path.unlink()
            print("✓ Cleaned up zip file")
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return False
    
    # Verify files
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    for file in expected_files:
        file_path = data_raw_path / file
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {file} ({file_size:.2f} MB)")
        else:
            print(f"✗ {file} - NOT FOUND")
    
    print("\n" + "=" * 60)
    print("Download complete! You can now run the EDA notebook.")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = download_m5_dataset()
    exit(0 if success else 1)
