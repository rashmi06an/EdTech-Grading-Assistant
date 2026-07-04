import os
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import joblib
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PORT = 8000
MODEL_PATH = "models/best_svm_model.joblib"
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.html")

# Global variables
model = None

def load_prediction_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Trained model not found at '{MODEL_PATH}'. "
                     "Please train the model first by running: python3 src/train.py")
        return False
    
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model successfully loaded from {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

class GradingAPIRequestHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        # Override to suppress standard HTTP access logging in terminal unless warning/error
        pass

    def do_GET(self):
        # Serve the drawing canvas front-end page
        if self.path == '/' or self.path == '/index.html':
            if not os.path.exists(INDEX_PATH):
                self.send_error(500, "Frontend index.html file not found in src/")
                return
                
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            
            with open(INDEX_PATH, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, f"File Not Found: {self.path}")

    def do_POST(self):
        # Handle predictions
        if self.path == '/predict':
            if model is None:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Classification model is not loaded on the server."}).encode('utf-8'))
                return

            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            try:
                data = json.loads(post_data)
                pixels = np.array(data['pixels'], dtype=np.float32)
                
                # Verify length
                if len(pixels) != 784:
                    logger.warning(f"Received malformed request: array size {len(pixels)} instead of 784.")
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"Invalid feature array size: expected 784, got {len(pixels)}"}).encode('utf-8'))
                    return
                
                # Perform model prediction
                # Reshape to (1, 784) for single sample inference
                features = pixels.reshape(1, -1)
                prediction = int(model.predict(features)[0])
                
                # Check for class probabilities
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features)[0].tolist()
                else:
                    # Fallback if probability estimates are disabled
                    # Apply Softmax to decision function scores to map to pseudo-probabilities
                    logger.info("Using Softmax fallback for decision function confidence mapping.")
                    decision_scores = model.decision_function(features)[0]
                    # Softmax formula to turn scores into probability distribution
                    exp_scores = np.exp(decision_scores - np.max(decision_scores))
                    probabilities = (exp_scores / np.sum(exp_scores)).tolist()
                
                # Form response
                response = {
                    "prediction": prediction,
                    "probabilities": probabilities
                }
                
                logger.info(f"API Inference: Graded digit as {prediction} | Confidence: {probabilities[prediction]*100:.1f}%")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                logger.error(f"Error handling /predict: {e}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Inference pipeline execution failed: {str(e)}"}).encode('utf-8'))
        else:
            self.send_error(404, f"API endpoint {self.path} not found.")

    def do_OPTIONS(self):
        # Support CORS for local development and direct index.html testing
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server():
    # Attempt to load model
    model_loaded = load_prediction_model()
    
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, GradingAPIRequestHandler)
    
    logger.info(f"--- Automated EdTech Grading Server ---")
    logger.info(f"Local Server URL: http://localhost:{PORT}")
    if not model_loaded:
        logger.warning("WARNING: Server started without a loaded model. Inference will fail until model is trained.")
    else:
        logger.info("System status: READY for grading classification.")
        
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down gracefully.")
    finally:
        httpd.server_close()

if __name__ == '__main__':
    run_server()
