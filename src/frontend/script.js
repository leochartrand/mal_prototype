// Global variables
let lossChart = null;

// Initialize Chart.js
function initChart() {
    const ctx = document.getElementById('loss-chart').getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    data: [],
                    borderColor: '#c88650',
                    backgroundColor: 'rgba(200, 134, 80, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Val Loss',
                    data: [],
                    borderColor: '#4caf50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#e0e0e0'
                    }
                },
                title: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += context.parsed.y.toFixed(6);
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Loss',
                        color: '#c88650'
                    },
                    ticks: {
                        color: '#b0b0b0'
                    },
                    grid: {
                        color: '#3a3a3a'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Epoch',
                        color: '#c88650'
                    },
                    ticks: {
                        color: '#b0b0b0'
                    },
                    grid: {
                        color: '#3a3a3a'
                    }
                }
            }
        }
    });
}

// Update progress bar
function updateProgress(elementId, textId, current, total) {
    const progressFill = document.getElementById(elementId);
    const progressText = document.getElementById(textId);
    
    const percentage = total > 0 ? (current / total) * 100 : 0;
    progressFill.style.width = percentage + '%';
    progressText.textContent = `${current}/${total}`;
}

// Update loss chart
function updateLossChart(data) {
    if (!lossChart || !data.history) return;
    
    const epochs = data.history.epochs || [];
    const trainLoss = data.history.train_loss || [];
    const valLoss = data.history.val_loss || [];
    
    lossChart.data.labels = epochs;
    lossChart.data.datasets[0].data = trainLoss;
    lossChart.data.datasets[1].data = valLoss;
    lossChart.update();
}

// Update loss table
function updateLossTable(data) {
    if (!data.losses) return;
    
    const losses = data.losses;
    
    // Update headers for component losses
    const headerRow = document.getElementById('loss-table-header');
    if (losses.train && losses.train.components) {
        const componentNames = Object.keys(losses.train.components);
        const componentHeaders = componentNames.map(name => `<th>${name}</th>`).join('');
        headerRow.innerHTML = `<th>Type</th><th>Total</th>${componentHeaders}`;
    }
    
    // Update train loss row
    const trainRow = document.getElementById('train-loss-row');
    if (losses.train) {
        const trainTotal = losses.train.total ? losses.train.total.toFixed(6) : '-';
        let trainHTML = `<td><strong>Train</strong></td><td>${trainTotal}</td>`;
        
        if (losses.train.components) {
            trainHTML += Object.values(losses.train.components)
                .map(val => `<td>${val.toFixed(6)}</td>`)
                .join('');
        }
        trainRow.innerHTML = trainHTML;
    }
    
    // Update val loss row
    const valRow = document.getElementById('val-loss-row');
    if (losses.val) {
        const valTotal = losses.val.total ? losses.val.total.toFixed(6) : '-';
        let valHTML = `<td><strong>Val</strong></td><td>${valTotal}</td>`;
        
        if (losses.val.components) {
            valHTML += Object.values(losses.val.components)
                .map(val => `<td>${val.toFixed(6)}</td>`)
                .join('');
        }
        valRow.innerHTML = valHTML;
    }
}

// Update visual outputs
function updateVisuals(data) {
    const visualGrid = document.getElementById('visual-grid');
    
    if (!data.visuals || !data.visuals.reconstructions || data.visuals.reconstructions.length === 0) {
        visualGrid.innerHTML = '<div class="placeholder">No visuals available yet</div>';
        return;
    }
    
    const images = data.visuals.reconstructions;
    visualGrid.innerHTML = images.map((imgPath, idx) => 
        `<img src="${imgPath}" alt="reconstruction ${idx}" />`
    ).join('');
}

// Update parameters display
function updateParameters(data) {
    const paramsContainer = document.getElementById('params-container');
    
    if (!data.parameters) {
        paramsContainer.innerHTML = '<p class="loading">No parameters available</p>';
        return;
    }
    
    const params = data.parameters;
    let html = '';
    
    // Filter out paths and unwanted keys
    const excludeKeys = ['dataset_path', 'model_path', 'results_path', 'checkpoint_path'];
    
    // Function to recursively flatten nested objects
    function flattenObject(obj, prefix = '') {
        const items = [];
        for (const [key, value] of Object.entries(obj)) {
            const fullKey = prefix ? `${prefix}.${key}` : key;
            
            if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                // Recursively flatten nested objects
                items.push(...flattenObject(value, fullKey));
            } else {
                items.push([fullKey, value]);
            }
        }
        return items;
    }
    
    const flatParams = flattenObject(params);
    
    // Just list all parameters in a clean format
    for (const [key, value] of flatParams) {
        // Skip excluded keys
        if (excludeKeys.some(excluded => key.includes(excluded))) continue;
        
        // Skip path-like values
        if (typeof value === 'string' && (value.includes('/') || value.includes('\\'))) continue;
        
        const displayValue = Array.isArray(value) ? 
            JSON.stringify(value) : String(value);
        
        html += `<div class="param-item">
            <span class="param-key">${key}</span>
            <span class="param-value">${displayValue}</span>
        </div>`;
    }
    
    paramsContainer.innerHTML = html || '<p class="loading">No parameters available</p>';
}

// Fetch and update all data
async function fetchData() {
    try {
        const response = await fetch('/api/training_data');
        const data = await response.json();
        
        // Update progress bars
        if (data.progress) {
            updateProgress('epoch-progress', 'epoch-text', 
                data.progress.current_epoch, data.progress.total_epochs);
            
            // Update batch progress with mode label
            const batchMode = data.progress.batch_mode || 'train';
            const batchLabel = batchMode === 'train' ? 'Batch Progress (Training)' : 'Batch Progress (Validation)';
            document.getElementById('batch-label').textContent = batchLabel;
            
            const batchProgressFill = document.getElementById('batch-progress');
            if (batchMode === 'val') {
                batchProgressFill.classList.add('val-mode');
            } else {
                batchProgressFill.classList.remove('val-mode');
            }
            
            updateProgress('batch-progress', 'batch-text',
                data.progress.current_batch, data.progress.total_batches);
        }
        
        // Update chart
        updateLossChart(data);
        
        // Update table
        updateLossTable(data);
        
        // Update visuals
        updateVisuals(data);
        
        // Update parameters (usually only once at start)
        updateParameters(data);
            
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

// Initialize
initChart();
fetchData();

// Auto-refresh every 2 seconds
setInterval(fetchData, 500);