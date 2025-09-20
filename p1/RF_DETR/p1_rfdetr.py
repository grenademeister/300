from file_management import prepare_dataset
from data_preprocessing import process_annotations
from core_training import train_rfdetr


def main():
    # Prepare dataset
    prepare_dataset()
    
    # Process annotations
    stats = process_annotations()
    
    # Check validation
    if not all(stats[split]["valid"] for split in stats):
        print("Annotation validation failed")
        return
    
    # Train model
    train_rfdetr()


if __name__ == "__main__":
    main()


