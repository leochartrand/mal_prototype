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
        self._csv_loaded = False
        
        # Initialize data structure
        self.data = {
            "progress": {
                "current_epoch": 0,
                "total_epochs": params["num_epochs"],
                "current_batch": 0,
                "total_batches": 0,
                "batch_mode": "train"  # 'train' or 'val'
            },
            "charts": {},     # populated by register_chart()
            "tables": {},     # populated by update_epoch()
            "visuals": {
                "reconstructions": [],
                "generations": []
            },
            "parameters": params,
        }
    
    def register_chart(self, name, series):
        """
        Register a named chart with series configuration.
        Call this before training starts for each chart you want displayed.
        
        Args:
            name: Chart title (e.g. "Generator", "Discriminator", "Loss")
            series: List of dicts with 'label' and 'color' keys, e.g.
                    [{"label": "Train", "color": "#c88650"},
                     {"label": "Val",   "color": "#b8bb26"}]
        """
        self.data["charts"][name] = {
            "order": len(self.data["charts"]),
            "series": series,
            "epochs": [],
            "history": {s["label"]: [] for s in series},
        }
        # Re-load CSV after each registration so all charts get populated
        self._csv_loaded = False
    
    def _load_history_from_csv(self):
        """Load training history from CSV file if it exists.
        Populates the first registered chart with total_loss per split,
        and creates history for any additional registered charts if matching
        CSV columns are found."""
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
                    
                    # Store all loss components dynamically
                    entry = {}
                    for key, val in row.items():
                        if key not in ('epoch', 'split') and val:
                            try:
                                entry[key] = float(val)
                            except (ValueError, TypeError):
                                pass
                    epoch_data[epoch][split] = entry
                
                # Map chart names to CSV columns
                # First registered chart defaults to total_loss;
                # others try <lowercase_name>_loss (e.g. "Discriminator" -> "disc_loss")
                chart_col_map = {}
                for i, cname in enumerate(self.data["charts"]):
                    if i == 0:
                        chart_col_map[cname] = 'total_loss'
                    else:
                        # "Discriminator" -> "disc_loss"
                        col = cname.lower()[:4] + '_loss'
                        chart_col_map[cname] = col

                # Build history lists from complete epochs (that have both train and val)
                for epoch in sorted(epoch_data.keys()):
                    if 'train' not in epoch_data[epoch] or 'val' not in epoch_data[epoch]:
                        continue
                    
                    for chart_name, chart in self.data["charts"].items():
                        col = chart_col_map.get(chart_name, 'total_loss')
                        train_val = epoch_data[epoch].get('train', {}).get(col)
                        val_val = epoch_data[epoch].get('val', {}).get(col)
                        # Skip epoch entirely for this chart if both splits lack data
                        if train_val is None and val_val is None:
                            continue
                        chart["epochs"].append(epoch)
                        for s in chart["series"]:
                            label = s["label"]
                            split = label.lower()
                            split_data = epoch_data[epoch].get(split, {})
                            chart["history"][label].append(split_data.get(col))
                
                # Set current_epoch to continue from last completed epoch
                complete_epochs = [e for e in epoch_data
                                   if 'train' in epoch_data[e] and 'val' in epoch_data[e]]
                if complete_epochs:
                    last_epoch = max(complete_epochs)
                    self.data["progress"]["current_epoch"] = last_epoch + 1
                    
                    # Populate tables with last epoch's component breakdown
                    last_train = epoch_data[last_epoch].get('train', {})
                    last_val = epoch_data[last_epoch].get('val', {})
                    
                    # Build a default table from CSV columns for the first chart
                    if self.data["charts"]:
                        first_chart = next(iter(self.data["charts"]))
                        self.data["tables"][first_chart] = {
                            "Train": {k: v for k, v in last_train.items()},
                            "Val": {k: v for k, v in last_val.items()},
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
        # Eagerly load CSV history so the frontend shows previous epochs
        # before the first new epoch completes (e.g. when resuming)
        if not self._csv_loaded:
            self._load_history_from_csv()
            self._csv_loaded = True

        self.data["progress"]["current_batch"] = batch_idx
        self.data["progress"]["batch_mode"] = mode
        if total_batches is not None:
            self.data["progress"]["total_batches"] = total_batches
    
    def update_epoch(self, epoch, charts=None, tables=None):
        """
        Update epoch-level data. Called once per epoch after all batches complete.
        
        Args:
            epoch: Current epoch number (0-indexed)
            charts: Dict mapping chart name -> {series_label: value}, e.g.
                     {"Generator": {"Train": 0.5, "Val": 0.4},
                      "Discriminator": {"Train": 0.3, "Val": 0.2}}
            tables: Dict mapping chart name -> {row_label: {col: val}}, e.g.
                     {"Generator": {
                         "Train": {"Flow Matching": 0.5, "Adversarial": 0.1},
                         "Val":   {"Flow Matching": 0.4}}}
        """
        charts = charts or {}
        tables = tables or {}
        
        # Lazy-load CSV history once all charts have been registered
        if not self._csv_loaded:
            self._load_history_from_csv()
            self._csv_loaded = True
        
        # Update chart histories
        for chart_name, values in charts.items():
            chart = self.data["charts"].get(chart_name)
            if chart is None:
                continue
            chart["epochs"].append(int(epoch))
            for s in chart["series"]:
                label = s["label"]
                v = values.get(label)
                chart["history"][label].append(float(v) if v is not None else None)
        
        # Update tables (current epoch display)
        for table_name, rows in tables.items():
            self.data["tables"][table_name] = {
                row_label: {k: float(v) if isinstance(v, (int, float)) else v
                            for k, v in cols.items()}
                for row_label, cols in rows.items()
            }
        
        # Load visuals from saved image (saved as epoch+1 in filename)
        self.load_visuals_from_path(epoch + 1)
        
        # Update progress bar to show completed epochs (epoch is 0-indexed)
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
    