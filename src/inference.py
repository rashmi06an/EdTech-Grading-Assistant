import os
import sys
import argparse
import joblib
import numpy as np

def ascii_art_digit(pixel_array):
    """
    Renders a 28x28 normalized pixel array as an ASCII art string for the terminal.
    
    Parameters:
    -----------
    pixel_array : numpy.ndarray
        Flat array of 784 features, scaled between 0.0 and 1.0.
        
    Returns:
    --------
    art : str
        ASCII representation of the digit image.
    """
    img = pixel_array.reshape(28, 28)
    # 10 levels of grayscale representation
    chars = " .:-=+*#%@"
    ascii_rows = []
    
    # We step by 2 in rows because console characters are typically taller than they are wide.
    # Stepping by 2 maintains a roughly square aspect ratio.
    for r in range(0, 28, 2):
        row_str = ""
        for c in range(28):
            val = img[r, c]
            char_idx = min(int(val * 9), 9)
            row_str += chars[char_idx]
        ascii_rows.append(row_str)
    return "\n".join(ascii_rows)

def load_and_preprocess_image(image_path):
    """
    Loads an image from disk, converts it to grayscale, resizes it to 28x28 pixels,
    and applies standard min-max scaling [0, 1]. Automatically inverts colors if
    it detects black ink on a white background.
    
    Parameters:
    -----------
    image_path : str
        Path to the custom image file.
        
    Returns:
    --------
    flat_img : numpy.ndarray
        Flattened 1D array of shape (784,) ready for model input.
    """
    try:
        from PIL import Image
    except ImportError:
        print("Error: The Pillow library is required to load custom image files.")
        print("Please install it using: pip install Pillow")
        sys.exit(1)
        
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    # Load and convert to grayscale ('L')
    img = Image.open(image_path).convert('L')
    # Resize to MNIST standard using Lanczos interpolation for high quality downsampling
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_arr = np.array(img)
    
    # Invert colors if needed. MNIST represents backgrounds as 0 (black) and strokes as 255 (white).
    # If the average intensity of the border pixels is bright, we assume black text on a white page
    # and invert the image so the model can classify it correctly.
    border_pixels = np.concatenate([
        img_arr[0, :],      # Top row
        img_arr[-1, :],     # Bottom row
        img_arr[:, 0],      # Left col
        img_arr[:, -1]      # Right col
    ])
    
    if np.mean(border_pixels) > 127:
        print("INFO: Bright background detected. Inverting colors for model compatibility.")
        img_arr = 255 - img_arr
        
    # Scale intensities to [0.0, 1.0]
    img_scaled = img_arr.astype(np.float32) / 255.0
    return img_scaled.flatten()

def main():
    parser = argparse.ArgumentParser(description="Automated EdTech Grading Assistant - Digit Inference Script")
    parser.add_argument("--model", type=str, default="models/best_svm_model.joblib", help="Path to the saved trained joblib model")
    parser.add_argument("--image", type=str, help="Path to a custom handwritten digit image file (PNG/JPG)")
    parser.add_argument("--demo", action="store_true", help="Run a command-line demo using a random digit from the dataset")
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        print("Please train the model first by running: python3 src/train.py")
        sys.exit(1)
        
    # Load model
    print(f"Loading classifier from {args.model}...")
    try:
        model = joblib.load(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    if args.image:
        # Predict custom image
        print(f"Loading custom image: {args.image}...")
        try:
            flat_features = load_and_preprocess_image(args.image)
            print("\nPreprocessed Digit Matrix (ASCII Art Rendering):")
            print(ascii_art_digit(flat_features))
            print("-" * 35)
            
            # Predict
            pred = model.predict([flat_features])[0]
            
            # Print confidence scores if model supports it
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([flat_features])[0]
                confidence = probs[pred] * 100
                print(f"Prediction: Digit '{pred}'")
                print(f"Confidence: {confidence:.2f}%")
                print("\nConfidence distribution:")
                for digit, p in enumerate(probs):
                    bar = "#" * int(p * 20)
                    print(f"  Digit {digit}: {p*100:5.1f}% | {bar}")
            else:
                print(f"Prediction: Digit '{pred}'")
                
        except Exception as e:
            print(f"Error executing prediction: {e}")
            sys.exit(1)
            
    elif args.demo:
        # Run demo on MNIST test segment
        print("Running demo classification using sample array...")
        try:
            from loader import load_mnist
            from scaler import scale_pixels_custom
            
            X, y = load_mnist(cast_to_int=True)
            X_scaled = scale_pixels_custom(X)
            
            # Pick a random sample index
            idx = np.random.randint(0, len(X_scaled))
            sample_features = X_scaled[idx]
            true_label = y[idx]
            
            print("\nSample Digit (ASCII Art Rendering):")
            print(ascii_art_digit(sample_features))
            print("-" * 35)
            
            pred = model.predict([sample_features])[0]
            print(f"True Label:       Digit '{true_label}'")
            print(f"Model Prediction: Digit '{pred}'")
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([sample_features])[0]
                print(f"Confidence:       {probs[pred]*100:.2f}%")
                
            if pred == true_label:
                print("Result:           SUCCESS! Correct prediction.")
            else:
                print("Result:           MISCLASSIFIED.")
                
        except Exception as e:
            print(f"Error running demo: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nNote: Specify either --image or --demo to run predictions.")

if __name__ == "__main__":
    main()
