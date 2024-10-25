import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata.single_table import SingleTableMetadata
import torch
from tqdm import tqdm
import argparse
import os
import sys
import json
from faker import Faker

# Initialize Faker for PII anonymization
fake = Faker()


def load_data(file_path):
    """
    Load data from a CSV or Excel file.

    Args:
        file_path (str): Path to the input file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    return df


def generate_metadata(df, table_name='synthetic_data', pii_columns=None):
    """
    Generate metadata for the DataFrame, handling PII columns if specified.

    Args:
        df (pd.DataFrame): The input DataFrame for which metadata is generated.
        table_name (str): The name to assign to the table in metadata.
        pii_columns (list): List of columns containing PII to be anonymized.

    Returns:
        Metadata: The generated metadata object.
        str: The table name.
    """
    # Initialize Metadata object with the dataframe
    metadata = Metadata.detect_from_dataframe(df, table_name=table_name)

    # Handle PII columns if specified
    if pii_columns:
        for column in pii_columns:
            if column in df.columns:
                # Update sdtype for PII columns
                new_sdtype = 'email' if 'email' in column.lower() else 'string'
                metadata.update_column(column_name=column, sdtype=new_sdtype, pii=True)
            else:
                print(f"Warning: PII column '{column}' not found in the data.")

    return metadata, table_name


def get_sdtype(series):
    """
    Determine the SDV data type based on pandas dtype and column content.

    Args:
        series (pd.Series): The pandas series for which to determine the sdtype.

    Returns:
        str: The SDV data type for the series.
    """
    dtype = str(series.dtype)
    
    if series.name and isinstance(series.name, str):
        column_name = series.name.lower()
        if 'email' in column_name:
            return 'email'
        elif 'date' in column_name or 'time' in column_name:
            return 'datetime'
        elif 'id' in column_name or 'code' in column_name:
            return 'id'
    
    if 'int' in dtype:
        return 'numerical'
    elif 'float' in dtype:
        return 'numerical'
    elif 'bool' in dtype:
        return 'boolean'
    elif 'datetime' in dtype:
        return 'datetime'
    elif 'object' in dtype or 'string' in dtype:
        # Check if it's an email by looking at the content
        if series.str.contains('@', na=False).mean() > 0.5:
            return 'email'
        return 'string'
    
    return 'string'  # default to string for unknown types


def generate_metadata_filename(base_path='metadata.json'):
    """
    Generate a unique metadata filename if the base filename already exists.

    Args:
        base_path (str): Base path for the metadata file.

    Returns:
        str: A unique metadata filename.
    """
    if not os.path.exists(base_path):
        return base_path
    
    # Split the filename and extension
    base_name, ext = os.path.splitext(base_path)
    counter = 1
    
    # Keep trying new filenames until we find one that doesn't exist
    while True:
        new_path = f"{base_name}_{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def save_metadata(metadata, metadata_path):
    """
    Save the metadata to a JSON file, handling existing files.

    Args:
        metadata (Metadata): The metadata object to save.
        metadata_path (str): Path to save the metadata file.

    Returns:
        str: The path where the metadata was saved.
    """
    try:
        # Try to save with the original filename
        new_path = generate_metadata_filename(metadata_path)
        metadata.save_to_json(new_path)
        print(f"Metadata saved to {new_path}")
        return new_path
    except Exception as e:
        print(f"Error saving metadata: {e}")
        raise


def load_metadata(metadata_path):
    """
    Load metadata from a JSON file.

    Args:
        metadata_path (str): Path to the metadata JSON file.

    Returns:
        Metadata: The loaded metadata object.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file '{metadata_path}' does not exist.")
    
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)
    
    metadata = Metadata()
    for table_name, table_info in metadata_dict['tables'].items():
        metadata.add_table(table_name, table_info)
    
    return metadata


def generate_synthetic_data(df, metadata, table_name, num_samples=100, epochs=300, batch_size=500, synthesizer_path=None, load_synthesizer_flag=False):
    """
    Generate synthetic data using SDV's CTGANSynthesizer with GPU support.

    Args:
        df (pd.DataFrame): The input DataFrame to train the synthesizer.
        metadata (Metadata): Metadata for the input DataFrame.
        table_name (str): Name of the table in metadata.
        num_samples (int): Number of synthetic samples to generate.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        synthesizer_path (str): Path to save/load the synthesizer model.
        load_synthesizer_flag (bool): Flag indicating whether to load an existing synthesizer model.

    Returns:
        pd.DataFrame: Generated synthetic data.
        CTGANSynthesizer: The trained synthesizer model.
    """
    # Determine device to use: GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the default device for PyTorch
    if device.type == 'cuda':
        torch.set_default_device(device)

    if load_synthesizer_flag and synthesizer_path and os.path.exists(synthesizer_path):
        synthesizer = CTGANSynthesizer.load(synthesizer_path)
        print(f"Synthesizer loaded from {synthesizer_path}")
    else:
        # Create synthesizer with the metadata
        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=epochs,
            batch_size=batch_size,
            verbose=True
        )

        print("Training synthesizer on", "GPU" if device.type == 'cuda' else "CPU")
        synthesizer.fit(df)

        if synthesizer_path:
            synthesizer.save(synthesizer_path)
            print(f"Synthesizer saved to {synthesizer_path}")

    synthetic_data = synthesizer.sample(num_samples)
    return synthetic_data, synthesizer


def anonymize_pii(df, pii_columns):
    """
    Anonymize specified PII columns using Faker.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be anonymized.
        pii_columns (list): List of PII columns to anonymize.

    Returns:
        pd.DataFrame: DataFrame with anonymized PII columns.
    """
    for column in pii_columns:
        if column in df.columns:
            if 'email' in column.lower():
                df[column] = [fake.email() for _ in range(len(df))]
            elif 'phone' in column.lower():
                df[column] = [fake.phone_number() for _ in range(len(df))]
            elif 'address' in column.lower():
                df[column] = [fake.address().replace('\n', ', ') for _ in range(len(df))]
            else:
                df[column] = [fake.word() for _ in range(len(df))]
    return df


def save_data(df, output_path):
    """
    Save the synthetic data to a CSV or Excel file.

    Args:
        df (pd.DataFrame): The DataFrame containing synthetic data to save.
        output_path (str): Path to save the synthetic data file.
    """
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith(('.xls', '.xlsx')):
        df.to_excel(output_path, index=False)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    print(f"Synthetic data saved to {output_path}")


def evaluate_synthetic_data(real_data, synthetic_data, metadata, report_path='quality_report.json'):
    """
    Evaluate the quality of the synthetic data and save the report.

    Args:
        real_data (pd.DataFrame): Original DataFrame containing real data.
        synthetic_data (pd.DataFrame): DataFrame containing generated synthetic data.
        metadata (Metadata): Metadata for the data.
        report_path (str): Path to save the quality report JSON file.

    Returns:
        QualityReport: The quality report for the generated synthetic data.
    """
    report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    
    # Assuming quality_report has a to_dict() method
    try:
        report_dict = report.to_dict()
        with open(report_path, 'w') as report_file:
            json.dump(report_dict, report_file, indent=4)
        print(f"Synthetic Data Quality Report saved to {report_path}")
    except AttributeError:
        print("Warning: Could not save report as a dictionary representation was not available.")
    
    return report


def main():
    """
    Main function to generate synthetic data from input CSV or Excel using SDV.

    Parses command-line arguments, loads data, handles PII columns, generates or loads metadata,
    trains or loads a synthesizer, generates synthetic data, anonymizes PII columns (if specified),
    and evaluates the quality of synthetic data.
    """
    parser = argparse.ArgumentParser(description="Generate Synthetic Data from CSV or Excel using SDV.")
    
    # Required arguments
    parser.add_argument('--input', required=True, help='Path to input CSV or Excel file.')
    parser.add_argument('--output', required=True, help='Path to save the synthetic CSV or Excel file.')
    
    # Optional arguments
    parser.add_argument('--num', type=int, default=100, help='Number of synthetic samples to generate.')
    parser.add_argument('--metadata', default='metadata.json', help='Path to save/load the metadata JSON file.')
    parser.add_argument('--pii', nargs='*', help='List of columns that contain PII to be anonymized or excluded.')
    parser.add_argument('--synthesizer', default='ctgan_synthesizer.pkl', help='Path to save/load the synthesizer model.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training the synthesizer.')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training the synthesizer.')
    parser.add_argument('--load_metadata', action='store_true', help='Load metadata from the specified metadata file instead of generating it.')
    parser.add_argument('--load_synthesizer', action='store_true', help='Load a pre-trained synthesizer from the specified synthesizer file.')
    parser.add_argument('--anonymize_pii', action='store_true', help='Anonymize PII columns in the generated synthetic data.')
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    num_samples = args.num
    metadata_path = args.metadata
    pii_columns = args.pii
    synthesizer_path = args.synthesizer
    epochs = args.epochs
    batch_size = args.batch_size
    load_metadata_flag = args.load_metadata
    load_synthesizer_flag = args.load_synthesizer
    anonymize_pii_flag = args.anonymize_pii
    
    # Define the steps for the progress bar
    steps = [
        "Loading data",
        "Handling PII columns",
        "Generating/Loading metadata",
        "Training/Loading synthesizer",
        "Generating synthetic data",
        "Anonymizing PII columns",
        "Saving synthetic data",
        "Evaluating synthetic data"
    ]
    
    # Initialize tqdm progress bar
    with tqdm(total=len(steps), desc="Overall Progress", unit="step") as pbar:
        
        # Step 1: Load the original data
        try:
            original_data = load_data(input_path)
            print("\nOriginal Data:")
            print(original_data.head())
            pbar.update(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
        
        # Step 2: Handle PII columns by excluding them from training
        if pii_columns:
            print("\nHandling PII columns...")
            # Exclude PII columns from synthetic data generation
            df_non_pii = original_data.copy()
            pbar.update(1)
        else:
            df_non_pii = original_data.copy()
            pbar.update(1)
        
        # Step 3: Generate or load metadata
        if args.load_metadata:
            try:
                metadata = load_metadata(args.metadata)
                table_name = list(metadata.tables.keys())[0]
                metadata_path = args.metadata
                pbar.update(1)
            except FileNotFoundError as e:
                print(e)
                sys.exit(1)
        else:
            print("\nGenerating metadata...")
            metadata, table_name = generate_metadata(original_data, table_name='synthetic_data', pii_columns=args.pii)
            metadata_path = save_metadata(metadata, args.metadata)
            pbar.update(1)
        
        # Step 4: Generate or load synthesizer and synthetic data
        if args.load_synthesizer and os.path.exists(args.synthesizer):
            synthesizer_path = args.synthesizer
        else:
            # Generate unique synthesizer filename based on metadata filename
            base_name = os.path.splitext(metadata_path)[0]
            synthesizer_path = f"{base_name}_synthesizer.pkl"
        
        synthetic_data, synthesizer = generate_synthetic_data(
            original_data,
            metadata,
            table_name,
            num_samples=args.num,
            epochs=args.epochs,
            batch_size=args.batch_size,
            synthesizer_path=synthesizer_path,
            load_synthesizer_flag=args.load_synthesizer
        )
        pbar.update(1)
        
        # Step 5: Anonymize PII columns if specified and flag is set
        if pii_columns and anonymize_pii_flag:
            print("\nAnonymizing PII columns...")
            synthetic_data = anonymize_pii(synthetic_data, pii_columns)
            print("PII columns have been anonymized.")
            print(synthetic_data.head())
            pbar.update(1)
        else:
            pbar.update(1)
        
        # Step 6: Save the synthetic data
        save_data(synthetic_data, output_path)
        pbar.update(1)
        
        # Step 7: Evaluate the synthetic data quality
        print("\nEvaluating synthetic data quality...")
        try:
            quality_report = evaluate_synthetic_data(
                original_data, 
                synthetic_data, 
                metadata, 
                report_path='quality_report.json'  # You can specify your desired path here
            )
            # Print the report if needed
            quality_report_dict = quality_report.to_dict()
            print("\nSynthetic Data Quality Report:")
            print(json.dumps(quality_report_dict, indent=4))
        except AttributeError:
            # If to_dict() is not available, print out some other representation
            print("\nSynthetic Data Quality Report:")
            print(quality_report)

        pbar.update(1)
    
    print("\nAll steps completed successfully!")


if __name__ == "__main__":
    main()
