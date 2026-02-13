// Gruvbox color palette (fallback defaults)
const GRUVBOX = {
    orange:     '#b86934',
    green:      '#b8bb26',
    purple:     '#d3869b',
    blue:       '#83a598',
    yellow:     '#fabd2f',
    red:        '#fb4934',
    aqua:       '#8ec07c',
    fg:         '#ebdbb2',
    fg_dim:     '#bdae93',
    bg:         '#282828',
    bg_soft:    '#3c3836',
    bg_hard:    '#1d2021',
    gray:       '#665c54',
};

// ─── Dynamic chart management ───────────────────────────────────────────────
// chartInstances[name] = Chart.js instance
const chartInstances = {};
// Track which chart sections have been created in the DOM
const createdSections = new Set();

// Chart.js shared options factory
function chartOptions(yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: {
            legend: {
                position: 'top',
                labels: { color: GRUVBOX.fg_dim, font: { size: 11 } }
            },
            tooltip: {
                callbacks: {
                    label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(6)}`
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: { display: true, text: yLabel, color: GRUVBOX.fg_dim },
                ticks: { color: GRUVBOX.fg_dim },
                grid: { color: GRUVBOX.gray }
            },
            x: {
                title: { display: true, text: 'Epoch', color: GRUVBOX.fg_dim },
                ticks: { color: GRUVBOX.fg_dim },
                grid: { color: GRUVBOX.gray }
            }
        }
    };
}

/**
 * Ensure a DOM section (canvas + table) exists for a given chart name.
 * Creates it on the first call, then returns the existing one.
 */
function ensureChartSection(name, seriesConfig, chartOrder) {
    if (createdSections.has(name)) return;
    createdSections.add(name);

    const grid = document.getElementById('grid-container');
    const visualSection = document.getElementById('visual-section');

    // Wrapper div
    const section = document.createElement('div');
    section.className = 'grid-item graph-item';
    section.id = `chart-section-${cssId(name)}`;
    // Use explicit registration order: chart 0 → order 1, chart 1 → order 3, etc.
    section.style.order = chartOrder * 2 + 1;

    // Title — use first series color for the accent
    const h3 = document.createElement('h3');
    h3.textContent = name;
    if (seriesConfig.length > 0) {
        h3.style.color = seriesConfig[0].color;
        h3.style.borderBottomColor = seriesConfig[0].color;
    }
    section.appendChild(h3);

    // Canvas
    const canvas = document.createElement('canvas');
    canvas.id = `chart-canvas-${cssId(name)}`;
    canvas.style.maxHeight = '300px';
    canvas.style.marginBottom = '15px';
    section.appendChild(canvas);

    // Table
    const tableWrap = document.createElement('div');
    tableWrap.className = 'loss-table-compact';
    const table = document.createElement('table');
    table.className = 'dynamic-loss-table';
    table.id = `chart-table-${cssId(name)}`;
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headerRow.id = `table-header-${cssId(name)}`;
    thead.appendChild(headerRow);
    const tbody = document.createElement('tbody');
    tbody.id = `table-body-${cssId(name)}`;
    table.appendChild(thead);
    table.appendChild(tbody);
    tableWrap.appendChild(table);
    section.appendChild(tableWrap);

    // Insert before visual section so charts fill the left column
    grid.insertBefore(section, visualSection);

    // Create Chart.js instance
    const datasets = seriesConfig.map(s => ({
        label: s.label,
        data: [],
        borderColor: s.color,
        backgroundColor: 'transparent',
        tension: 0.3,
        borderWidth: 2,
    }));
    chartInstances[name] = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels: [], datasets },
        options: chartOptions('Loss'),
    });
}

/** Sanitize a chart name for use as a CSS id */
function cssId(name) {
    return name.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();
}

// ─── Update helpers ─────────────────────────────────────────────────────────

function updateProgress(elementId, textId, current, total) {
    const progressFill = document.getElementById(elementId);
    const progressText = document.getElementById(textId);
    const percentage = total > 0 ? (current / total) * 100 : 0;
    progressFill.style.width = percentage + '%';
    progressText.textContent = `${current}/${total}`;
}

/** Update all registered charts from data.charts */
function updateCharts(data) {
    if (!data.charts) return;
    for (const [name, chartData] of Object.entries(data.charts)) {
        const chartOrder = chartData.order !== undefined ? chartData.order : 0;
        ensureChartSection(name, chartData.series || [], chartOrder);
        const chart = chartInstances[name];
        if (!chart) continue;

        chart.data.labels = chartData.epochs || [];
        (chartData.series || []).forEach((s, i) => {
            chart.data.datasets[i].data = chartData.history[s.label] || [];
        });
        chart.update();
    }
}

/** Update all tables from data.tables */
function updateTables(data) {
    if (!data.tables) return;
    for (const [name, rows] of Object.entries(data.tables)) {
        const headerEl = document.getElementById(`table-header-${cssId(name)}`);
        const bodyEl = document.getElementById(`table-body-${cssId(name)}`);
        if (!headerEl || !bodyEl) continue;

        // Collect all column keys across all rows, with "Total" first
        const colSet = new Set();
        for (const cols of Object.values(rows)) {
            for (const k of Object.keys(cols)) colSet.add(k);
        }
        const colNames = [];
        if (colSet.has('Total')) { colNames.push('Total'); colSet.delete('Total'); }
        colNames.push(...colSet);

        // Header
        headerEl.innerHTML = '<th></th>' + colNames.map(c => `<th>${c}</th>`).join('');

        // Rows — use series colors from the matching chart if available
        const chartData = data.charts && data.charts[name];
        const seriesColors = {};
        if (chartData && chartData.series) {
            for (const s of chartData.series) seriesColors[s.label] = s.color;
        }

        let bodyHtml = '';
        for (const [rowLabel, cols] of Object.entries(rows)) {
            const color = seriesColors[rowLabel] || GRUVBOX.fg;
            bodyHtml += '<tr>';
            bodyHtml += `<td style="color:${color}"><strong>${rowLabel}</strong></td>`;
            for (const col of colNames) {
                const v = cols[col];
                bodyHtml += `<td>${typeof v === 'number' ? v.toFixed(6) : (v ?? '-')}</td>`;
            }
            bodyHtml += '</tr>';
        }
        bodyEl.innerHTML = bodyHtml;
    }
}

/** Update visual outputs */
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

/** Update parameters display */
function updateParameters(data) {
    const paramsContainer = document.getElementById('params-container');
    if (!data.parameters) {
        paramsContainer.innerHTML = '<p class="loading">No parameters available</p>';
        return;
    }
    const params = data.parameters;
    const excludeKeys = ['dataset_path', 'model_path', 'results_path', 'checkpoint_path'];

    function flattenObject(obj, prefix = '') {
        const items = [];
        for (const [key, value] of Object.entries(obj)) {
            const fullKey = prefix ? `${prefix}.${key}` : key;
            if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                items.push(...flattenObject(value, fullKey));
            } else {
                items.push([fullKey, value]);
            }
        }
        return items;
    }

    const flatParams = flattenObject(params);
    let html = '';
    for (const [key, value] of flatParams) {
        if (excludeKeys.some(ex => key.includes(ex))) continue;
        if (typeof value === 'string' && (value.includes('/') || value.includes('\\'))) continue;
        const displayValue = Array.isArray(value) ? JSON.stringify(value) : String(value);
        html += `<div class="param-item">
            <span class="param-key">${key}</span>
            <span class="param-value">${displayValue}</span>
        </div>`;
    }
    paramsContainer.innerHTML = html || '<p class="loading">No parameters available</p>';
}

// ─── Main fetch loop ────────────────────────────────────────────────────────

async function fetchData() {
    try {
        const response = await fetch('/api/training_data');
        const data = await response.json();

        // Progress
        if (data.progress) {
            updateProgress('epoch-progress', 'epoch-text',
                data.progress.current_epoch, data.progress.total_epochs);
            const batchMode = data.progress.batch_mode || 'train';
            document.getElementById('batch-label').textContent =
                batchMode === 'train' ? 'Batch Progress (Training)' : 'Batch Progress (Validation)';
            const batchFill = document.getElementById('batch-progress');
            batchFill.classList.toggle('val-mode', batchMode === 'val');
            updateProgress('batch-progress', 'batch-text',
                data.progress.current_batch, data.progress.total_batches);
        }

        updateCharts(data);
        updateTables(data);
        updateVisuals(data);
        updateParameters(data);
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

// Start polling
fetchData();
setInterval(fetchData, 500);