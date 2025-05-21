from ultralytics import YOLO

# Ensure the code runs only when the script is executed directly, not when imported
if __name__ == '__main__':
    # Load the pre-trained mode
    model = YOLO("yolo11n.pt")

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data="dataset/data.yaml",  # Path to dataset configuration file
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image size for training
        device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )

    # Validate the model after training
    metrics = model.val()
