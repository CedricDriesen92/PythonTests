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


document.getElementById('file-upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);

    try {
        const response = await fetch('/process-file', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.error) {
            alert(data.error);
        } else {
            gridData = data;
            initializeGrid();
            document.getElementById('grid-editor').classList.remove('hidden');
            document.getElementById('pathfinder').classList.remove('hidden');
            updateFloorDisplay();
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing the file.');
    }
});

function initializeGrid() {
    const container = document.getElementById('grid-container');
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const gridWidth = gridData.grids[0][0].length;
    const gridHeight = gridData.grids[0].length;

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
    renderGrid(gridData.grids[currentFloor]);
}

function renderGrid(grid) {
    const container = document.getElementById('grid-container');
    container.innerHTML = '';
    container.style.display = 'grid';
    container.style.gridTemplateColumns = `repeat(${grid[0].length}, ${cellSize}px)`;

    grid.forEach((row, i) => {
        row.forEach((cell, j) => {
            const div = document.createElement('div');
            div.style.width = `${cellSize}px`;
            div.style.height = `${cellSize}px`;
            div.classList.add('border', 'border-gray-300');
            div.dataset.row = i;
            div.dataset.col = j;

            updateCellAppearance(div, cell);

            div.addEventListener('mousedown', startPainting);
            div.addEventListener('mousemove', handleMouseMove);
            div.addEventListener('mouseup', stopPainting);
            container.appendChild(div);
        });
    });

    container.addEventListener('mouseleave', handleMouseLeave);
}

function updateCellAppearance(cellElement, cellType) {
    cellElement.className = 'border border-gray-300';
    if (cellElement.classList.contains('preview')) {
        cellElement.classList.add('preview');
    }
    cellElement.style.width = `${cellSize}px`;
    cellElement.style.height = `${cellSize}px`;
    switch (cellType) {
        case 'wall':
            cellElement.classList.add('bg-black');
            break;
        case 'door':
            cellElement.classList.add('bg-orange-500');
            break;
        case 'stair':
            cellElement.classList.add('bg-red-500');
            break;
        case 'floor':
            cellElement.classList.add('bg-pink-100');
            break;
        default:
            cellElement.classList.add('bg-white');
    }
}

function startPainting(e) {
    isPainting = true;
    isMouseDown = true;
    lastPaintedCell = null;
    clearPreview();
    paint(e);
}

function stopPainting() {
    isPainting = false;
    isMouseDown = false;
    lastPreviewCell = lastPaintedCell;
    lastPaintedCell = null;
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
        renderGrid(gridData.grids[currentFloor]);
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
    if (row >= 0 && row < gridData.grids[currentFloor].length &&
        col >= 0 && col < gridData.grids[currentFloor][0].length) {
        const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
        if (cell) {
            cell.classList.add('preview');
            updateCellAppearance(cell, currentType);
            previewCells.add(cell);
        }
    }
}

function clearPreview() {
    previewCells.forEach(cell => {
        cell.classList.remove('preview');
        const row = parseInt(cell.dataset.row);
        const col = parseInt(cell.dataset.col);
        updateCellAppearance(cell, gridData.grids[currentFloor][row][col]);
    });
    previewCells.clear();
}

function handleMouseMove(e) {
    if (isPainting) {
        paint(e);
    } else {
        showPreview(e);
    }
}

function handleMouseLeave() {
    clearPreview();
    stopPainting();
}

document.getElementById('next-floor').addEventListener('click', () => {
    if (currentFloor < gridData.grids.length - 1) {
        currentFloor++;
        renderGrid(gridData.grids[currentFloor]);
        updateFloorDisplay();
    }
});


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

// Prevent dragging on the grid container
document.getElementById('grid-container').addEventListener('dragstart', (e) => e.preventDefault());

// Stop painting when mouse leaves the grid
document.getElementById('grid-container').addEventListener('mouseleave', stopPainting);

document.getElementById('download-grid').addEventListener('click', downloadGrid);