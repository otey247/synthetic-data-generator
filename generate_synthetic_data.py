import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.evaluation.single_table import evaluate_quality
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
    """
    metadata = Metadata()
    # Create a dictionary of DataFrames
    tables = {
        table_name: df  # Pass the DataFrame directly
    }
    # Detect metadata from dictionary of tables
    metadata.detect_from_dataframes(tables)
    
    if pii_columns:
        for column in pii_columns:
            if column in df.columns:
                metadata.update_column_property(table_name, column, 'pii', True)
            else:
                print(f"Warning: PII column '{column}' not found in the data.")
    
    return metadata

def save_metadata(metadata, metadata_path):
    """
    Save the metadata to a JSON file for future use.
    """
    metadata.save_to_json(metadata_path)
    print(f"Metadata saved to {metadata_path}")

def load_metadata(metadata_path):
    """
    Load metadata from a JSON file.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file '{metadata_path}' does not exist.")
    metadata = Metadata.load_json(metadata_path)
    print(f"Metadata loaded from {metadata_path}")
    return metadata

def anonymize_pii(df, pii_columns):
    """
    Anonymize specified PII columns using Faker.
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

def generate_synthetic_data(df, metadata, num_samples=100, epochs=300, batch_size=500, synthesizer_path=None, load_synthesizer_flag=False):
    """
    Generate synthetic data using SDV's CTGANSynthesizer.
    Optionally load a pre-trained synthesizer if synthesizer_path is provided.
    """
    # Get the table name from metadata
    table_name = list(metadata.tables.keys())[0]
    
    if load_synthesizer_flag and synthesizer_path and os.path.exists(synthesizer_path):
        synthesizer = CTGANSynthesizer.load(synthesizer_path)
        print(f"Synthesizer loaded from {synthesizer_path}")
    else:
        synthesizer = CTGANSynthesizer(
            metadata.tables[table_name], 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=True
        )
        print("Training synthesizer...")
        synthesizer.fit(df)
        if synthesizer_path:
            synthesizer.save(synthesizer_path)
            print(f"Synthesizer saved to {synthesizer_path}")
    
    synthetic_data = synthesizer.sample(num_samples)
    return synthetic_data, synthesizer

def save_data(df, output_path):
    """
    Save the synthetic data to a CSV or Excel file.
    """
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith(('.xls', '.xlsx')):
        df.to_excel(output_path, index=False)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    print(f"Synthetic data saved to {output_path}")

def evaluate_synthetic_data(real_data, synthetic_data, metadata):
    """
    Evaluate the quality of the synthetic data.
    """
    report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    return report

def main():
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
        if load_metadata_flag:
            try:
                metadata = load_metadata(metadata_path)
                pbar.update(1)
            except FileNotFoundError as e:
                print(e)
                sys.exit(1)
        else:
            print("\nGenerating metadata...")
            metadata = generate_metadata(original_data, table_name='synthetic_data', pii_columns=pii_columns)
            save_metadata(metadata, metadata_path)
            pbar.update(1)
        
        # Step 4: Generate or load synthesizer and synthetic data
        synthetic_data, synthesizer = generate_synthetic_data(
            original_data,
            metadata,
            num_samples=num_samples,
            epochs=epochs,
            batch_size=batch_size,
            synthesizer_path=synthesizer_path,
            load_synthesizer_flag=load_synthesizer_flag
        )
        pbar.update(1)
        
        # Step 5: Anonymize PII columns if specified
        if pii_columns:
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
        
        # Step 7: Evaluate the synthetic data
        print("\nEvaluating synthetic data quality...")
        quality_report = evaluate_synthetic_data(original_data, synthetic_data, metadata)
        print("\nSynthetic Data Quality Report:")
        print(json.dumps(quality_report, indent=4))
        pbar.update(1)
    
    print("\nAll steps completed successfully!")

if __name__ == "__main__":
    main()