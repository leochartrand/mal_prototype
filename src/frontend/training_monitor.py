import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import threading
import time
import atexit

# Global reference to monitor instance for Flask to access
_monitor_instance = None

# Flask server will be started in background
_server_thread = None
_server_started = False


def _start_flask_server():
    """Start Flask server in background thread"""
    global _server_started
    if _server_started:
        return
    
    try:
        from flask import Flask, jsonify, send_from_directory
        from flask_cors import CORS
        
        app = Flask(__name__, static_folder=str(Path(__file__).parent))
        CORS(app)
        
        frontend_dir = Path(__file__).parent
        
        @app.route('/')
        def index():
            return send_from_directory(frontend_dir, 'index.html')
        
        @app.route('/<path:path>')
        def serve_static(path):
            return send_from_directory(frontend_dir, path)
        
        @app.route('/api/training_data')
        def get_training_data():
            # Serve data directly from monitor instance in memory
            global _monitor_instance
            return jsonify(_monitor_instance.data)
        
        @app.route('/images/<path:path>')
        def serve_images(path):
            images_dir = Path(__file__).parent.parent.parent / 'results'
            return send_from_directory(images_dir, path)
        
        # Run server with logging disabled
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        print("=" * 70)
        print("🚀 Training Monitor Frontend Started!")
        print(f"📊 View at: http://localhost:8080")
        print("=" * 70)
        
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True, use_reloader=False)
        
    except Exception as e:
        print(f"Warning: Could not start Flask server: {e}")
        print("Training will continue without web monitoring.")


def _init_server():
    """Initialize Flask server in background thread"""
    global _server_thread, _server_started
    if not _server_started:
        _server_thread = threading.Thread(target=_start_flask_server, daemon=True)
        _server_thread.start()
        _server_started = True
        time.sleep(1)  # Give server a moment to start


class TrainingMonitor:
    def __init__(self, params):
        """
        Initialize the training monitor and start Flask server.
        
        Args:
            params: Dictionary of model parameters
        """
        # Register this instance globally so Flask can access it
        global _monitor_instance
        _monitor_instance = self
        
        # Start Flask server in background
        _init_server()
        
        self.params = params
        self.last_mode = None
        self.last_batch_idx = -1
        
        # Initialize data structure
        self.data = {
            "progress": {
                "current_epoch": 0,
                "total_epochs": params["num_epochs"],
                "current_batch": 0,
                "total_batches": 0,
                "batch_mode": "train"  # 'train' or 'val'
            },
            "history": {
                "epochs": [],
                "train_loss": [],
                "val_loss": []
            },
            "losses": {
                "train": {"total": None, "components": {}},
                "val": {"total": None, "components": {}}
            },
            "visuals": {
                "reconstructions": [],
                "generations": []
            },
            "parameters": params,
        }
        
        # Load existing training history from CSV if available
        self._load_history_from_csv()
    
    def _load_history_from_csv(self):
        """Load training history from CSV file if it exists"""
        if 'results_path' not in self.params:
            return
        
        import csv
        log_file = Path(self.params['results_path']) / 'training_log.csv'
        
        if not log_file.exists():
            return
        
        try:
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                epoch_data = {}
                
                for row in reader:
                    epoch = int(row['epoch'])
                    split = row['split']
                    
                    if epoch not in epoch_data:
                        epoch_data[epoch] = {}
                    
                    # Store all loss components
                    epoch_data[epoch][split] = {
                        'total_loss': float(row['total_loss']),
                        'recon_loss': float(row.get('recon_loss', 0)),
                        'vq_loss': float(row.get('vq_loss', 0)),
                        'commit_loss': float(row.get('commit_loss', 0)),
                        'diversity_loss': float(row.get('diversity_loss', 0)),
                        'confidence_loss': float(row.get('confidence_loss', 0)),
                    }
                
                # Build history lists from complete epochs (that have both train and val)
                for epoch in sorted(epoch_data.keys()):
                    if 'train' in epoch_data[epoch] and 'val' in epoch_data[epoch]:
                        self.data["history"]["epochs"].append(epoch)
                        self.data["history"]["train_loss"].append(epoch_data[epoch]['train']['total_loss'])
                        self.data["history"]["val_loss"].append(epoch_data[epoch]['val']['total_loss'])
                
                # Set current_epoch to continue from last completed epoch
                # And populate current loss display with last epoch's values
                if self.data["history"]["epochs"]:
                    last_epoch = self.data["history"]["epochs"][-1]
                    self.data["progress"]["current_epoch"] = last_epoch + 1
                    
                    # Set current losses to last epoch's values
                    last_train = epoch_data[last_epoch]['train']
                    last_val = epoch_data[last_epoch]['val']
                    
                    self.data["losses"]["train"]["total"] = last_train['total_loss']
                    self.data["losses"]["train"]["components"] = {
                        'recon_loss': last_train['recon_loss'],
                        'vq_loss': last_train['vq_loss'],
                        'commit_loss': last_train['commit_loss'],
                        'diversity_loss': last_train['diversity_loss'],
                        'confidence_loss': last_train['confidence_loss'],
                    }
                    
                    self.data["losses"]["val"]["total"] = last_val['total_loss']
                    self.data["losses"]["val"]["components"] = {
                        'recon_loss': last_val['recon_loss'],
                        'vq_loss': last_val['vq_loss'],
                        'commit_loss': last_val['commit_loss'],
                        'diversity_loss': last_val['diversity_loss'],
                        'confidence_loss': last_val['confidence_loss'],
                    }
                    
                    # Load visual from last completed epoch (add 1 for filename format)
                    self.load_visuals_from_path(last_epoch + 1)
        except Exception as e:
            print(f"[Monitor] Warning: Could not load history from CSV: {e}")
    
    def update_batch(self, batch_idx, total_batches=None, mode="train"):
        """
        Update current batch progress
        
        Args:
            batch_idx: Current batch index
            total_batches: Total number of batches (optional)
            mode: 'train' or 'val'
        """
        self.data["progress"]["current_batch"] = batch_idx
        self.data["progress"]["batch_mode"] = mode
        if total_batches is not None:
            self.data["progress"]["total_batches"] = total_batches
    
    def update_epoch(self, epoch, train_loss, train_components, val_loss, val_components):
        """
        Update epoch-level average losses, graph, and visuals.
        Called once per epoch after all batches are complete.
        
        Args:
            epoch: Current epoch number (0-indexed, from training loop)
            train_loss: Average training loss for the epoch
            train_components: Dict of average component losses for training
            val_loss: Average validation loss for the epoch  
            val_components: Dict of average component losses for validation
        """
        
        # Update display losses
        self.data["losses"]["train"]["total"] = float(train_loss)
        self.data["losses"]["train"]["components"] = {
            k: float(v) for k, v in train_components.items()
        }
        self.data["losses"]["val"]["total"] = float(val_loss)
        self.data["losses"]["val"]["components"] = {
            k: float(v) for k, v in val_components.items()
        }
        
        # Update history for graph
        self.data["history"]["epochs"].append(int(epoch))
        self.data["history"]["train_loss"].append(float(train_loss))
        self.data["history"]["val_loss"].append(float(val_loss))
        
        # Load visuals from saved image (saved as epoch+1 in filename)
        self.load_visuals_from_path(epoch + 1)
        
        # Update progress bar to show completed epochs (epoch is 0-indexed, so epoch+1 = number completed)
        self.data["progress"]["current_epoch"] = epoch + 1
    
    def load_visuals_from_path(self, epoch, model_name=None):
        """
        Load existing visualization from results/{model_name}/decoded/epoch_n.png
        
        Args:
            epoch: Epoch number matching the saved filename (already adjusted for 1-indexing)
            model_name: Model name (if None, extracts from results_path in params)
        """
        if model_name is None and 'results_path' in self.params:
            # Extract model name from results_path like 'results/flexvae/'
            results_path = self.params['results_path']
            # Remove 'results/' prefix and trailing slash
            if results_path.startswith('results/'):
                model_name = results_path[8:].rstrip('/')
            else:
                model_name = results_path.rstrip('/')
        
        if model_name:
            # Look for the image at results/{model_name}/decoded/epoch_XXX.png
            # Epoch parameter should already match the saved filename format
            image_path = f"{model_name}/decoded/epoch_{epoch:03d}.png"
            full_path = Path(__file__).parent.parent.parent / 'results' / image_path
            
            if full_path.exists():
                # Path relative to /images/ endpoint which serves from results/
                vis_path = f"/images/{image_path}"
                self.data["visuals"]["reconstructions"] = [vis_path]
                self.data["visuals"]["generations"] = [vis_path]
    