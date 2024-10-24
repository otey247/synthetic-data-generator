# Synthetic Data Generator

A Python script for generating synthetic data from CSV or Excel files using SDV (Synthetic Data Vault). This tool supports PII handling, metadata management, and customizable synthetic data generation parameters.

## Features

- Generate synthetic data from CSV or Excel files
- Anonymize PII (Personally Identifiable Information) columns
- Save and load metadata for consistency
- Save and load trained synthesizers for reuse
- Evaluate synthetic data quality
- Progress tracking with detailed feedback
- Customizable synthesis parameters

## Installation

1. Clone the repository:
```bash
git clone https://github.com/OteyJo/synthetic-data-generator.git
cd synthetic-data-generator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Basic Usage

### Generate Simple Synthetic Data
Generate 100 synthetic samples from your data:
```bash
python generate_synthetic_data.py --input your_data.csv --output synthetic_data.csv
```

### Handle PII Columns
Anonymize specific columns containing personal information:
```bash
python generate_synthetic_data.py --input your_data.csv --output synthetic_data.csv --pii Email PhoneNumber
```

### Customize Generation Parameters
Adjust sample size, epochs, and batch size:
```bash
python generate_synthetic_data.py \
    --input your_data.csv \
    --output synthetic_data.csv \
    --num 200 \
    --epochs 500 \
    --batch_size 1000
```

## Advanced Usage

### Save and Load Metadata
```bash
# Save metadata
python generate_synthetic_data.py \
    --input your_data.csv \
    --output synthetic_data.csv \
    --metadata my_metadata.json

# Load existing metadata
python generate_synthetic_data.py \
    --input your_data.csv \
    --output synthetic_data.csv \
    --metadata my_metadata.json \
    --load_metadata
```

### Save and Reuse Synthesizer
```bash
# Train and save synthesizer
python generate_synthetic_data.py \
    --input your_data.csv \
    --output synthetic_data.csv \
    --synthesizer my_ctgan.pkl

# Load existing synthesizer
python generate_synthetic_data.py \
    --input your_data.csv \
    --output synthetic_data.csv \
    --synthesizer my_ctgan.pkl \
    --load_synthesizer
```

### Combined Example
Use multiple features together:
```bash
python generate_synthetic_data.py \
    --input your_data.csv \
    --output synthetic_data.csv \
    --num 150 \
    --metadata my_metadata.json \
    --pii Email PhoneNumber \
    --synthesizer my_ctgan.pkl \
    --epochs 400 \
    --batch_size 800 \
    --load_metadata \
    --load_synthesizer
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input CSV or Excel file | Required |
| `--output` | Path to save the synthetic CSV or Excel file | Required |
| `--num` | Number of synthetic samples to generate | 100 |
| `--metadata` | Path to save/load the metadata JSON file | metadata.json |
| `--pii` | List of columns containing PII to be anonymized | None |
| `--synthesizer` | Path to save/load the synthesizer model | ctgan_synthesizer.pkl |
| `--epochs` | Number of epochs for training | 300 |
| `--batch_size` | Batch size for training | 500 |
| `--load_metadata` | Load existing metadata file | False |
| `--load_synthesizer` | Load existing synthesizer | False |

## Output

The script provides detailed progress information and generates:
1. Synthetic data file (CSV/Excel)
2. Metadata file (JSON)
3. Synthesizer model file (PKL)
4. Quality evaluation report

The quality report includes metrics for:
- Data validity
- Data consistency
- Statistical similarity
- Overall quality score

## Dependencies

- pandas
- sdv
- faker
- tqdm
- argparse
