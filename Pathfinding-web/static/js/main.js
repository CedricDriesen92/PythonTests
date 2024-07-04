let gridData = null;
let currentFloor = 0;
let currentType = 'wall';
let currentTool = 'paint';
let start = null;
let goals = [];
let cellSize = 20;
let isPainting = false;
let minZoom = 1;
let brushSize = 1;
let lastPaintedCell = null;
let lastPreviewCell = null;
let previewCells = new Set();
let isMouseDown = false;
let wallBuffer = 0;
let paintedCells = new Set();


function uploadFile(event) {
    event.preventDefault();
    const formData = new FormData(event.target);

    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');

    progressContainer.classList.remove('hidden');
    progressBar.style.width = '0%';
    progressText.textContent = 'Initializing...';

    // First, send the file
    fetch('/process-file', {
        method: 'POST',
        body: formData
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        // If file upload successful, start listening for progress
        const eventSource = new EventSource('/process-file-sse');

        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.progress !== undefined) {
                const progress = data.progress;
                progressBar.style.width = `${progress}%`;
                progressText.textContent = `${progress.toFixed(1)}%: ${data.message}`;
            } else if (data.complete) {
                eventSource.close();
                progressContainer.classList.add('hidden');
                // Handle the completed data
                gridData = data.result;
                initializeGrid();
                document.getElementById('grid-editor').classList.remove('hidden');
                document.getElementById('pathfinder').classList.remove('hidden');
                updateFloorDisplay();
            } else if (data.error) {
                eventSource.close();
                progressContainer.classList.add('hidden');
                console.error('Processing error:', data.error);
                alert(`An error occurred while processing the file: ${data.error}`);
            }
        };

        eventSource.onerror = function(error) {
            console.error('EventSource failed:', error);
            eventSource.close();
            progressContainer.classList.add('hidden');
            alert('An error occurred while processing the file.');
        };
    }).catch(error => {
        console.error('Error:', error);
        progressContainer.classList.add('hidden');
        alert('An error occurred while uploading the file.');
    });
}

document.getElementById('file-upload-form').addEventListener('submit', uploadFile);

function initializeGrid() {
    const container = document.getElementById('grid-container');
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const gridWidth = gridData.grids[0][0].length;
    const gridHeight = gridData.grids[0].length;
    // After loading initial grid data
    gridData.buffered_grids = gridData.grids;
    updateBufferForPaintedCells();

    console.log(`Container: ${containerWidth}x${containerHeight}, Grid: ${gridWidth}x${gridHeight}`);

    minZoom = Math.max(1, Math.min(containerWidth / gridWidth, containerHeight / gridHeight));
    cellSize = minZoom;

    console.log(`minZoom: ${minZoom}, initial cellSize: ${cellSize}`);

    const zoomSlider = document.getElementById('zoom-slider');
    zoomSlider.min = 100;  // Minimum 10% zoom
    zoomSlider.max = 2000; // Maximum 200% zoom
    zoomSlider.value = 1000; // Start at 100% zoom
    zoomPercentage = zoomSlider.value;
    cellSize = (zoomPercentage / 100) * minZoom;

    updateZoomLevel();
    renderGrid(gridData.buffered_grids[currentFloor]);
}

function renderGrid(grid) {
    const container = document.getElementById('grid-container');
    container.innerHTML = '';

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const gridWidth = grid[0].length;
    const gridHeight = grid.length;

    canvas.width = gridWidth * cellSize;
    canvas.height = gridHeight * cellSize;

    // Set canvas size
    canvas.style.width = `${gridWidth * cellSize}px`;
    canvas.style.height = `${gridHeight * cellSize}px`;

    // Draw grid
    grid.forEach((row, i) => {
        row.forEach((cell, j) => {
            ctx.fillStyle = getCellColor(cell);
            ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);

            // Draw cell border
            //ctx.strokeStyle = '#e5e7eb'; // border-gray-300 equivalent
            //ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);
        });
    });

    container.appendChild(canvas);

    // Add event listeners
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseLeave);
}

function getCellColor(cellType) {
    switch (cellType) {
        case 'wall': return '#000000';
        case 'door': return '#f97316';
        case 'stair': return '#ef4444';
        case 'floor': return '#fbcfe8';
        case 'walla': return '#9ca3af';
        default: return '#ffffff';
    }
}

function updateCellAppearance(row, col, cellType) {
    const canvas = document.querySelector('#grid-container canvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = getCellColor(cellType);
    ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
    ctx.strokeStyle = '#e5e7eb';
    ctx.strokeRect(col * cellSize, row * cellSize, cellSize, cellSize);
}

function startPainting(e) {
    isPainting = true;
    isMouseDown = true;
    lastPaintedCell = null;
    clearPreview();
    paint(e);
}

function stopPainting() {
    gridData.buffered_grids = gridData.grids;
    isPainting = false;
    isMouseDown = false;
    lastPreviewCell = lastPaintedCell;
    lastPaintedCell = null;
    if (paintedCells.size > 0) {
        updateBufferForPaintedCells();
    }
}

function paint(e) {
    clearPreview();
    const row = parseInt(e.target.dataset.row);
    const col = parseInt(e.target.dataset.col);

    if (currentType === 'start') {
        start = { floor: currentFloor, row, col };
        renderGrid(gridData.grids[currentFloor]);
    } else if (currentType === 'goal') {
        goals.push({ floor: currentFloor, row, col });
        renderGrid(gridData.grids[currentFloor]);
    } else if (currentTool === 'fill') {
        floodFill(currentFloor, row, col, gridData.grids[currentFloor][row][col]);
        renderGrid(gridData.grids[currentFloor]);
    } else {
        paintWithBrush(row, col);
        if (lastPaintedCell) {
            interpolatePaint(lastPaintedCell.row, lastPaintedCell.col, row, col);
        }
    }

    lastPaintedCell = { row, col };
    showPreview(e);
}

function paintWithBrush(centerRow, centerCol) {
    const halfSize = Math.floor(brushSize / 2);
    for (let i = 0; i < brushSize; i++) {
        for (let j = 0; j < brushSize; j++) {
            const row = centerRow - halfSize + i;
            const col = centerCol - halfSize + j;
            if (row >= 0 && row < gridData.grids[currentFloor].length &&
                col >= 0 && col < gridData.grids[currentFloor][0].length) {
                gridData.grids[currentFloor][row][col] = currentType;
                paintedCells.add(`${row},${col}`);
                const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                if (cell) {
                    updateCellAppearance(cell, currentType);
                }
            }
        }
    }
}

function interpolatePaint(startRow, startCol, endRow, endCol) {
    const dx = Math.abs(endCol - startCol);
    const dy = Math.abs(endRow - startRow);
    const sx = startCol < endCol ? 1 : -1;
    const sy = startRow < endRow ? 1 : -1;
    let err = dx - dy;

    while (true) {
        paintWithBrush(startRow, startCol);

        if (startRow === endRow && startCol === endCol) break;
        const e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            startCol += sx;
        }
        if (e2 < dx) {
            err += dx;
            startRow += sy;
        }
    }
}

function floodFill(floor, row, col, targetElement) {
    if (row < 0 || row >= gridData.grids[floor].length ||
        col < 0 || col >= gridData.grids[floor][0].length ||
        gridData.grids[floor][row][col] !== targetElement ||
        targetElement === currentType) {
        return;
    }
    tempSize = brushSize;
    paintWithBrush(row, col);
    brushSize = tempSize;
    floodFill(floor, row + 1, col, targetElement);
    floodFill(floor, row - 1, col, targetElement);
    floodFill(floor, row, col + 1, targetElement);
    floodFill(floor, row, col - 1, targetElement);
}

document.getElementById('draw-wall').addEventListener('click', () => {currentType = 'wall'; currentTool = 'paint'});
document.getElementById('draw-door').addEventListener('click', () => {currentType = 'door'; currentTool = 'paint'});
document.getElementById('draw-stair').addEventListener('click', () => {currentType = 'stair'; currentTool = 'paint'});
document.getElementById('draw-floor').addEventListener('click', () => {currentType = 'floor'; currentTool = 'paint'});
document.getElementById('draw-empty').addEventListener('click', () => {currentType = 'empty'; currentTool = 'paint'});

document.getElementById('prev-floor').addEventListener('click', () => {
    if (currentFloor > 0) {
        currentFloor--;
        updateBufferForPaintedCells();
        renderGrid(gridData.buffered_grids[currentFloor]);
        updateFloorDisplay();
    }
});


document.getElementById('next-floor').addEventListener('click', () => {
    if (currentFloor < gridData.grids.length - 1) {
        currentFloor++;
        updateBufferForPaintedCells();
        renderGrid(gridData.buffered_grids[currentFloor]);
        updateFloorDisplay();
    }
});

function showPreview(e) {
    if (currentTool !== 'paint' || currentType === 'start' || currentType === 'goal') return;

    const row = parseInt(e.target.dataset.row);
    const col = parseInt(e.target.dataset.col);

    clearPreview();

    // Show brush preview
    previewBrush(row, col);

    // Show interpolation preview only if brush size is 1 and there's a last preview cell
    if (brushSize === 1 && lastPreviewCell) {
        previewInterpolation(lastPreviewCell.row, lastPreviewCell.col, row, col);
    }
    lastPreviewCell = { row, col };
}

function previewBrush(centerRow, centerCol) {
    const halfSize = Math.floor(brushSize / 2);
    for (let i = 0; i < brushSize; i++) {
        for (let j = 0; j < brushSize; j++) {
            const previewRow = centerRow - halfSize + i;
            const previewCol = centerCol - halfSize + j;
            previewCell(previewRow, previewCol);
        }
    }
}

function previewInterpolation(startRow, startCol, endRow, endCol) {
    const dx = Math.abs(endCol - startCol);
    const dy = Math.abs(endRow - startRow);
    const sx = startCol < endCol ? 1 : -1;
    const sy = startRow < endRow ? 1 : -1;
    let err = dx - dy;
    let currentRow = startRow;
    let currentCol = startCol;

    while (true) {
        previewBrush(currentRow, currentCol);

        if (currentRow === endRow && currentCol === endCol) break;
        const e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            currentCol += sx;
        }
        if (e2 < dx) {
            err += dx;
            currentRow += sy;
        }
    }
}

function previewCell(row, col) {
    const canvas = document.querySelector('#grid-container canvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = getCellColor(currentType);
    ctx.globalAlpha = 0.5;
    ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
    ctx.globalAlpha = 1.0;
}

function clearPreview() {
    renderGrid(gridData.buffered_grids[currentFloor]);
}

function handleMouseDown(e) {
    isMouseDown = true;
    const { row, col } = getCellCoordinates(e);
    startPainting({ target: { dataset: { row, col } } });
}

function handleMouseMove(e) {
    const { row, col } = getCellCoordinates(e);
    if (isMouseDown) {
        paint({ target: { dataset: { row, col } } });
    } else {
        showPreview({ target: { dataset: { row, col } } });
    }
}

function handleMouseUp() {
    stopPainting();
}

function handleMouseLeave(){
    stopPainting();
}

function getCellCoordinates(e) {
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);
    return { row, col };
}


document.getElementById('fill-tool').addEventListener('click', () => currentTool = 'fill');

document.getElementById('clear-floor').addEventListener('click', () => {
    gridData.grids[currentFloor] = gridData.grids[currentFloor].map(row => row.map(() => 'empty'));
    renderGrid(gridData.grids[currentFloor]);
});

document.getElementById('add-floor').addEventListener('click', () => {
    const newFloor = gridData.grids[currentFloor].map(row => row.map(() => 'empty'));
    gridData.grids.push(newFloor);
    gridData.floors.push({
        elevation: gridData.floors[gridData.floors.length - 1].elevation + gridData.floors[gridData.floors.length - 1].height,
        height: gridData.floors[gridData.floors.length - 1].height
    });
    currentFloor = gridData.grids.length - 1;
    renderGrid(gridData.grids[currentFloor]);
    updateFloorDisplay();
});

document.getElementById('remove-floor').addEventListener('click', () => {
    if (gridData.grids.length > 1) {
        gridData.grids.pop();
        gridData.floors.pop();
        currentFloor = Math.min(currentFloor, gridData.grids.length - 1);
        renderGrid(gridData.grids[currentFloor]);
        updateFloorDisplay();
    } else {
        alert('Cannot remove the last floor.');
    }
});

document.getElementById('brush-size').addEventListener('input', (e) => {
    brushSize = parseInt(e.target.value);
    document.getElementById('brush-size-display').textContent = brushSize;
});

document.getElementById('zoom-slider').addEventListener('input', (e) => {
    const zoomPercentage = parseInt(e.target.value);
    cellSize = (zoomPercentage / 100) * minZoom;
    console.log(`Zoom slider value: ${zoomPercentage}, new cellSize: ${cellSize}`);
    updateZoomLevel();
    renderGrid(gridData.grids[currentFloor]);
});

function updateZoomLevel() {
    const zoomPercentage = Math.round((cellSize / minZoom) * 100);
    console.log(`Updating zoom level: cellSize=${cellSize}, minZoom=${minZoom}, zoomPercentage=${zoomPercentage}`);
    document.getElementById('zoom-level').textContent = `${zoomPercentage/10}%`;
}

function updateFloorDisplay() {
    document.getElementById('current-floor').textContent = `Floor: ${currentFloor + 1} / ${gridData.grids.length}`;
}

document.getElementById('set-start').addEventListener('click', () => currentType = 'start');
document.getElementById('set-goal').addEventListener('click', () => currentType = 'goal');

document.getElementById('find-path').addEventListener('click', async () => {
    if (!start || goals.length === 0) {
        alert('Please set start and at least one goal.');
        return;
    }

    try {
        const response = await fetch('/find-path', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                grids: gridData.grids,
                grid_size: gridData.grid_size,
                floors: gridData.floors,
                bbox: gridData.bbox,
                start: start,
                goals: goals
            })
        });
        const data = await response.json();
        document.getElementById('result').innerHTML = `<pre>${JSON.stringify(data.path, null, 2)}</pre>`;

        highlightPath(data.path);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while finding the path.');
    }
});

function highlightPath(path) {
    path.forEach(point => {
        if (point[2] === currentFloor) {
            const cell = document.querySelector(`[data-row="${point[0]}"][data-col="${point[1]}"]`);
            if (cell) {
                cell.classList.add('bg-yellow-300');
            }
        }
    });
}

function downloadGrid() {
    if (!gridData) {
        alert('No grid data available. Please upload or create a grid first.');
        return;
    }

    const dataToSave = {
        grids: gridData.grids,
        grid_size: gridData.grid_size,
        floors: gridData.floors,
        bbox: gridData.bbox
    };

    const jsonString = JSON.stringify(dataToSave, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const downloadLink = document.createElement('a');
    downloadLink.href = url;
    downloadLink.download = 'edited_grid.json';

    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);

    URL.revokeObjectURL(url);
}

document.getElementById('wall-buffer').addEventListener('input', (e) => {
    wallBuffer = parseInt(e.target.value);
    document.getElementById('wall-buffer-display').textContent = wallBuffer;
    updateWallBuffer(wallBuffer);
});

function updateBufferForPaintedCells() {
    //const affectedCells = Array.from(paintedCells).map(coord => coord.split(',').map(Number));
    affectedCells = [0];
    fetch('/update-buffer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            floor: currentFloor,
            affected_cells: affectedCells,
            wall_buffer: wallBuffer,
            grids: gridData.grids,
            grid_size:gridData.grid_size,
            floors:gridData.floors,
            bbox:gridData.bbox
        })
    })
    .then(response => response.json())
    .then(data => {
        gridData.buffered_grids[currentFloor] = data.updated_floor;
        renderGrid(gridData.buffered_grids[currentFloor]);
        paintedCells.clear();
    })
    .catch(error => console.error('Error:', error));
}

function updateWallBuffer(newBufferValue) {
    fetch('/apply-wall-buffer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            grids: gridData.grids,
            wall_buffer: newBufferValue
        })
    })
    .then(response => response.json())
    .then(data => {
        gridData.buffered_grids = data.buffered_grids;
        renderGrid(gridData.buffered_grids[currentFloor]);
    })
    .catch(error => console.error('Error:', error));
}

// Prevent dragging on the grid container
document.getElementById('grid-container').addEventListener('dragstart', (e) => e.preventDefault());

// Stop painting when mouse leaves the grid
document.getElementById('grid-container').addEventListener('mouseleave', stopPainting);

document.getElementById('download-grid').addEventListener('click', downloadGrid);

