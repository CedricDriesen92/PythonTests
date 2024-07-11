// State variables
let originalGridData = null;
let bufferedGridData = null;
let currentFloor = 0;
let currentType = 'wall';
let currentTool = 'paint';
let start = null;
let goals = [];
let cellSize = 20;
let isPainting = false;
let minZoom = 1;
let zoomLevel = 10000;
let brushSize = 1;
let lastPaintedCell = null;
let lastPreviewCell = null;
let previewCells = new Set();
let isMouseDown = false;
let wallBuffer = 0;
let paintedCells = new Set();
let pathfindingSteps = [];
let currentStepIndex = 0;
let animationInterval = null;
let pathData = null;
let allowDiagonal = true;
let minimizeCost = true;
let spacesData = [];
let includeEmptyTiles = false;
let foundEscapeRoutes = null;
let spaceColors = {};
let maxStairDistance = 30;

// DOM elements
const gridContainer = document.getElementById('grid-container');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const fileUploadForm = document.getElementById('file-upload-form');
const zoomSlider = document.getElementById('zoom-slider');
const brushSizeSlider = document.getElementById('brush-size');
const wallBufferSlider = document.getElementById('wall-buffer');
const allowDiagonalCheckbox = document.getElementById('allow-diagonal');
const minimizeCostCheckbox = document.getElementById('minimize-cost');
const exportPathButton = document.getElementById('export-path');
const detectExitsButton = document.getElementById('detect-exits');
const spaceDetectionButton = document.getElementById('update-spaces');
const maxStairDistanceSlider = document.getElementById('max-stair-distance');
const maxStairDistanceDisplay = document.getElementById('max-stair-distance-display');
const outputText = document.getElementById('result');

// Event listeners
fileUploadForm.addEventListener('submit', uploadFile);
zoomSlider.addEventListener('input', handleZoomChange);
brushSizeSlider.addEventListener('input', handleBrushSizeChange);
wallBufferSlider.addEventListener('input', handleWallBufferChange);
gridContainer.addEventListener('mousedown', handleMouseDown);
gridContainer.addEventListener('mousemove', handleMouseMove);
gridContainer.addEventListener('mouseup', handleMouseUp);
gridContainer.addEventListener('mouseleave', handleMouseLeave);
gridContainer.addEventListener('mousemove', handleGridHover);
gridContainer.addEventListener('mouseout', handleGridMouseOut);
gridContainer.addEventListener('dragstart', (e) => e.preventDefault());
allowDiagonalCheckbox.addEventListener('change', (e) => {
allowDiagonal = e.target.checked;
});
minimizeCostCheckbox.addEventListener('change', (e) => {
    minimizeCost = e.target.checked;
});
exportPathButton.addEventListener('click', exportPath);
gridContainer.addEventListener('contextmenu', (e) => e.preventDefault());
detectExitsButton.addEventListener('click', detectExits);
document.getElementById('include-empty-tiles').addEventListener('change', (e) => {
    includeEmptyTiles = e.target.checked;
});
spaceDetectionButton.addEventListener('click', updateSpaces);
maxStairDistanceSlider.addEventListener('input', handleMaxStairDistanceChange);

document.getElementById('draw-wall').addEventListener('click', () => setCurrentType('wall'));
document.getElementById('draw-door').addEventListener('click', () => setCurrentType('door'));
document.getElementById('draw-stair').addEventListener('click', () => setCurrentType('stair'));
document.getElementById('draw-floor').addEventListener('click', () => setCurrentType('floor'));
document.getElementById('draw-empty').addEventListener('click', () => setCurrentType('empty'));
document.getElementById('fill-tool').addEventListener('click', () => setCurrentTool('fill'));
document.getElementById('prev-floor').addEventListener('click', navigateToPreviousFloor);
document.getElementById('next-floor').addEventListener('click', navigateToNextFloor);
document.getElementById('clear-floor').addEventListener('click', clearCurrentFloor);
document.getElementById('add-floor').addEventListener('click', addNewFloor);
document.getElementById('remove-floor').addEventListener('click', removeCurrentFloor);
document.getElementById('set-start').addEventListener('click', () => setCurrentType('start'));
document.getElementById('set-goal').addEventListener('click', () => setCurrentType('goal'));
document.getElementById('find-path').addEventListener('click', findPath);
document.getElementById('download-grid').addEventListener('click', downloadGrid);
document.getElementById('calculate-escape-routes').addEventListener('click', calculateEscapeRoutes);



async function uploadFile(event) {
    event.preventDefault();
    const formData = new FormData(event.target);

    showProgress('Initializing...');

    try {
        const response = await fetch('/api/process-file', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        console.log('Received data from server:', result);
        handleProcessedData(result);
    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred while processing the file.');
    } finally {
        hideProgress();
    }
}

function handleProcessedData(data) {
    originalGridData = data;
    bufferedGridData = {...data};
    spacesData = data.spaces || [];
    console.log('Loaded spaces data:', spacesData);
    
    if (typeof originalGridData.grid_size === 'number') {
        originalGridData.grid_size *= originalGridData.unit_size || 1;
        bufferedGridData.grid_size *= bufferedGridData.unit_size || 1;
        console.log('Adjusted grid size:', bufferedGridData.grid_size);
    } else {
        console.error('Invalid grid_size in loaded data:', originalGridData.grid_size);
        showError('Invalid grid size in loaded data');
        return;
    }

    if (!Array.isArray(originalGridData.grids) || originalGridData.grids.length === 0) {
        console.error('Invalid grids data in loaded file:', originalGridData.grids);
        showError('Invalid grids data in loaded file');
        return;
    }

    console.log('Grid data seems valid, initializing grid...');
    initializeGrid();
    showGridEditor();
}

function initializeGrid() {
    console.log('Initializing grid...');
    const containerWidth = gridContainer.clientWidth;
    const containerHeight = gridContainer.clientHeight;
    console.log('Container dimensions:', containerWidth, containerHeight);

    const gridWidth = bufferedGridData.grids[0][0].length;
    const gridHeight = bufferedGridData.grids[0].length;
    console.log('Grid dimensions:', gridWidth, gridHeight);

    const horizontalZoom = containerWidth / gridWidth;
    const verticalZoom = containerHeight / gridHeight;
    minZoom = Math.min(horizontalZoom, verticalZoom);
    console.log('Calculated zoom levels:', horizontalZoom, verticalZoom, minZoom);

    zoomLevel = 20*Math.max(100, Math.min(25000, Math.round(minZoom * 100)));
    cellSize = (zoomLevel / 100) * bufferedGridData.grid_size;
    console.log('Set zoom level and cell size:', zoomLevel, cellSize);

    zoomSlider.value = zoomLevel;
    updateZoomLevel();
    renderGrid(bufferedGridData.grids[currentFloor]);
    updateFloorDisplay();
}

function renderGrid(grid) {
    //console.log('Rendering grid for floor:', currentFloor);
    //console.log('Grid data:', grid);
    
    gridContainer.innerHTML = '';
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = grid[0].length * cellSize;
    canvas.height = grid.length * cellSize;
    //console.log('Canvas dimensions:', canvas.width, canvas.height);

    // Render the base grid
    grid.forEach((row, i) => {
        row.forEach((cell, j) => {
            ctx.fillStyle = getCellColor(cell);
            ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
        });
    });

    // Render IFC spaces
    renderSpaces(ctx);

    // Render path if it exists
    if (pathData) {
        console.log(pathData);
        ctx.fillStyle = 'rgba(255, 255, 0, 0.5)';
        pathData.forEach(point => {
            if (point[2] === currentFloor) {
                ctx.fillRect(point[1] * cellSize, point[0] * cellSize, cellSize, cellSize);
            }
        });
    }

    if (foundEscapeRoutes) {
        let totalLength = 0;
        let stairwayDistance = 0;
        let spacesOverMaxDistance = [];
        foundEscapeRoutes.forEach(route => {
            totalLength = Math.max(totalLength, route.distance);
            stairwayDistance = Math.max(stairwayDistance, route.distance_to_stair);
            const isOverMaxDistance = route.distance_to_stair > maxStairDistance;
            if (isOverMaxDistance){
                spacesOverMaxDistance.push(route.space_name);
                outputText.innerHTML.concat("\nSpace: ", route.space_name, "\nDistance to stairway too long: ", route.distance_to_stair, "\nTotal route length: ", route.distance);
            }
            ctx.strokeStyle = isOverMaxDistance ? 'rgba(255, 0, 0, 0.8)' : 'rgba(0, 255, 0, 0.8)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            route.optimal_path.forEach((point, index) => {
                if (point[2] === currentFloor) {
                    const x = (point[1] + 0.5) * cellSize;
                    const y = (point[0] + 0.5) * cellSize;
                    if (index === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
            });
            ctx.stroke();

            // Mark the furthest point
            const furthestPoint = route.furthest_point;
            if (furthestPoint[2] === currentFloor) {
                ctx.fillStyle = 'purple';
                const x = (furthestPoint[1] + 0.5) * cellSize;
                const y = (furthestPoint[0] + 0.5) * cellSize;
                ctx.beginPath();
                ctx.arc(x, y, cellSize * 2, 0, 2 * Math.PI);
                ctx.fill();
            }
        });
        const pathLengthsElement = document.getElementById('path-lengths');
        pathLengthsElement.innerHTML = `
            <h3>Escape Route Lengths:</h3>
            <p>Longest escape route: ${totalLength.toFixed(2)} meters</p>
            <p>Longest distance to stairway: ${stairwayDistance.toFixed(2)} meters</p>
            <p>Spaces over max stair distance: ${spacesOverMaxDistance.join(', ') || 'None'}</p>
        `;
    }


    // Render start point if it exists and is on the current floor
    if (start && start.floor === currentFloor) {
        ctx.fillStyle = 'green';
        ctx.beginPath();
        ctx.arc((start.col + 0.5) * cellSize, (start.row + 0.5) * cellSize, cellSize * 2, 0, 2 * Math.PI);
        ctx.fill();
    }

    // Render goal points if they exist and are on the current floor
    if (goals.length > 0) {
        ctx.fillStyle = 'red';
        goals.forEach(goal => {
            if (goal.floor === currentFloor) {
                ctx.beginPath();
                ctx.arc((goal.col + 0.5) * cellSize, (goal.row + 0.5) * cellSize, cellSize * 2, 0, 2 * Math.PI);
                ctx.fill();
            }
        });
    }


    gridContainer.appendChild(canvas);
    //console.log('Grid rendered and appended to container');
}

function renderSpaces(ctx) {
    const currentFloorSpaces = spacesData.filter(space => space.floor === currentFloor);

    console.log('Rendering spaces for floor:', currentFloor, ' number: ', currentFloorSpaces.length);
    
    currentFloorSpaces.forEach((space, index) => {
        if (!spaceColors[space.id]) {
            spaceColors[space.id] = generateRandomColor();
        }
        
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.fillStyle = spaceColors[space.id];
        
        if (!space.polygon || space.polygon.length === 0) {
            console.warn(`Space ${space.id} has no polygon`);
            return;
        }

        ctx.beginPath();
        space.polygon.forEach((point, i) => {
            const x = ((point[0] - bufferedGridData.bbox.min_x) / bufferedGridData.grid_size + 0.5) * cellSize;
            const y = ((point[1] - bufferedGridData.bbox.min_y) / bufferedGridData.grid_size + 0.5) * cellSize;
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // Render space name and path lengths
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.font = '12px Arial';
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const centerX = space.polygon.reduce((sum, p) => sum + p[0], 0) / space.polygon.length;
        const centerY = space.polygon.reduce((sum, p) => sum + p[1], 0) / space.polygon.length;
        const textX = ((centerX - bufferedGridData.bbox.min_x) / bufferedGridData.grid_size + 0.5) * cellSize;
        const textY = ((centerY - bufferedGridData.bbox.min_y) / bufferedGridData.grid_size + 0.5) * cellSize;
        
        ctx.fillText(space.name, textX, textY);
        
        if (foundEscapeRoutes) {
            const route = foundEscapeRoutes.find(r => r.id === space.id);
            if (route) {
                ctx.fillText(`Total: ${route.distance.toFixed(2)}m`, textX, textY + 15);
                ctx.fillText(`To Stair: ${route.distance_to_stair.toFixed(2)}m`, textX, textY + 30);
            }
        }
    });
}


async function updateSpaces() {
    try {
        const response = await fetch('/api/update-spaces', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                include_empty_tiles: includeEmptyTiles,
                grids: bufferedGridData.grids,
                grid_size: bufferedGridData.grid_size,
                floors: bufferedGridData.floors,
                bbox: bufferedGridData.bbox,
            })
        });
        const data = await response.json();
        if (response.ok) {
            spacesData = data.spaces;
            renderGrid(bufferedGridData.grids[currentFloor]);
            //showMessage('Spaces updated successfully');
        } else {
            showError(`Space update error: ${data.error}`);
        }
    } catch (error) {
        console.error('Error:', error);
        showError(`An error occurred while updating spaces: ${error.message}`);
    }
}

function getCellColor(cellType) {
    const colors = {
        'wall': '#000000',
        'door': '#f97316',
        'stair': '#ef4444',
        'floor': '#fbcfe8',
        'walla': '#9ca3af',
        'empty': '#ffffff'
    };
    return colors[cellType] || '#ffffff';
}

function handleMouseDown(e) {
    if (e.button === 2) { // Right click
        e.preventDefault();
        const { row, col } = getCellCoordinates(e);
        removeStartOrGoal(row, col);
    } else {
        isMouseDown = true;
        const { row, col } = getCellCoordinates(e);
        startPainting(row, col);
    }
}

function removeStartOrGoal(row, col) {
    if (start && start.row === row && start.col === col && start.floor === currentFloor) {
        start = null;
    } else {
        goals = goals.filter(goal => !(goal.row === row && goal.col === col && goal.floor === currentFloor));
    }
    renderGrid(bufferedGridData.grids[currentFloor]);
}

function handleMouseMove(e) {
    const { row, col } = getCellCoordinates(e);
    if (isMouseDown) {
        paint(row, col);
    } else {
        showPreview(row, col);
    }
}

function handleMouseUp() {
    stopPainting();
}

function handleMouseLeave() {
    stopPainting();
}

function handleGridHover(e) {
    const { row, col } = getCellCoordinates(e);
    const space = spacesData.find(s => s.floor === currentFloor && isPointInPolygon(row, col, s.polygon));
    if (space && foundEscapeRoutes) {
        const route = foundEscapeRoutes.find(r => r.id === space.id);
        if (route) {
            highlightPath(route.optimal_path);
        }
    }
}

function handleGridMouseOut() {
    renderGrid(bufferedGridData.grids[currentFloor]);
}

function isPointInPolygon(x, y, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = (polygon[i][0] - bufferedGridData.bbox.min_x) / bufferedGridData.grid_size;
        const yi = (polygon[i][1] - bufferedGridData.bbox.min_y) / bufferedGridData.grid_size;
        const xj = (polygon[j][0] - bufferedGridData.bbox.min_x) / bufferedGridData.grid_size;
        const yj = (polygon[j][1] - bufferedGridData.bbox.min_y) / bufferedGridData.grid_size;

        const intersect = ((yi > y) !== (yj > y))
            && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

function highlightPath(path) {
    const canvas = gridContainer.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    
    // Redraw the grid
    renderGrid(bufferedGridData.grids[currentFloor]);
    
    // Highlight the path
    ctx.strokeStyle = 'rgba(255, 255, 0, 0.8)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    path.forEach((point, index) => {
        if (point[2] === currentFloor) {
            const x = (point[1] + 0.5) * cellSize;
            const y = (point[0] + 0.5) * cellSize;
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
    });
    ctx.stroke();
}

function startPainting(row, col) {
    isPainting = true;
    lastPaintedCell = null;
    clearPreview();
    paint(row, col);
}

function stopPainting() {
    isPainting = false;
    isMouseDown = false;
    lastPreviewCell = lastPaintedCell;
    lastPaintedCell = null;
    if (paintedCells.size > 0) {
        const updates = Array.from(paintedCells).map(coord => {
            const [row, col] = coord.split(',').map(Number);
            return { floor: currentFloor, row, col, type: originalGridData.grids[currentFloor][row][col] };
        });
        batchUpdateCells(updates).then(() => {
            paintedCells.clear();
            renderGrid(bufferedGridData.grids[currentFloor]);
        });
    }
}

function paint(row, col) {
    clearPreview();

    if (currentType === 'start') {
        start = { floor: currentFloor, row, col };
    } else if (currentType === 'goal') {
        goals.push({ floor: currentFloor, row, col });
    } else if (currentTool === 'fill') {
        floodFill(currentFloor, row, col, originalGridData.grids[currentFloor][row][col]);
    } else {
        paintWithBrush(row, col);
        if (lastPaintedCell) {
            interpolatePaint(lastPaintedCell.row, lastPaintedCell.col, row, col);
        }
    }

    lastPaintedCell = { row, col };
    renderGrid(bufferedGridData.grids[currentFloor]);
    showPreview(row, col);
}

function paintWithBrush(centerRow, centerCol) {
    const updates = [];
    const halfSize = Math.floor(brushSize / 2);
    for (let i = 0; i < brushSize; i++) {
        for (let j = 0; j < brushSize; j++) {
            const row = centerRow - halfSize + i;
            const col = centerCol - halfSize + j;
            if (isValidCell(row, col)) {
                if (originalGridData.grids[currentFloor][row][col] != currentType){
                    originalGridData.grids[currentFloor][row][col] = currentType;
                    bufferedGridData.grids[currentFloor][row][col] = currentType;
                    updates.push({ floor: currentFloor, row, col, type: currentType });
                    paintedCells.add(`${row},${col}`);
                }
            }
        }
    }
    return updates;
}

function interpolatePaint(startRow, startCol, endRow, endCol) {
    const updates = [];
    const dx = Math.abs(endCol - startCol);
    const dy = Math.abs(endRow - startRow);
    const sx = startCol < endCol ? 1 : -1;
    const sy = startRow < endRow ? 1 : -1;
    let err = dx - dy;

    while (true) {
        updates.push(...paintWithBrush(startRow, startCol));

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
    return updates;
}

function floodFill(floor, row, col, targetElement) {
    const updates = [];
    const stack = [[row, col]];
    const seen = new Set();

    while (stack.length > 0) {
        const [r, c] = stack.pop();
        const key = `${r},${c}`;
        if (seen.has(key)) continue;
        seen.add(key);

        if (!isValidCell(r, c) || originalGridData.grids[floor][r][c] !== targetElement || targetElement === currentType) {
            continue;
        }

        originalGridData.grids[floor][r][c] = currentType;
        updates.push({ floor, row: r, col: c, type: currentType });

        stack.push([r+1, c], [r-1, c], [r, c+1], [r, c-1]);
    }

    return updates;
}

function showPreview(row, col) {
    if (currentTool !== 'paint' || currentType === 'start' || currentType === 'goal') return;

    clearPreview();
    previewBrush(row, col);

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
    if (!isValidCell(row, col)) return;
    const canvas = gridContainer.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = getCellColor(currentType);
    ctx.globalAlpha = 0.5;
    ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
    ctx.globalAlpha = 1.0;
}

function clearPreview() {
    renderGrid(bufferedGridData.grids[currentFloor]);
}

function getCellCoordinates(e) {
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);
    return { row, col };
}

function isValidCell(row, col) {
    return row >= 0 && row < bufferedGridData.grids[currentFloor].length &&
           col >= 0 && col < bufferedGridData.grids[currentFloor][0].length;
}

function setCurrentType(type) {
    currentType = type;
    currentTool = 'paint';
}

function setCurrentTool(tool) {
    currentTool = tool;
}

function navigateToPreviousFloor() {
    if (currentFloor > 0) {
        currentFloor--;
        renderGrid(bufferedGridData.grids[currentFloor]);
        updateFloorDisplay();
    }
}

function navigateToNextFloor() {
    if (currentFloor < bufferedGridData.grids.length - 1) {
        currentFloor++;
        renderGrid(bufferedGridData.grids[currentFloor]);
        updateFloorDisplay();
    }
}

async function clearCurrentFloor() {
    const updates = [];
    for (let row = 0; row < originalGridData.grids[currentFloor].length; row++) {
        for (let col = 0; col < originalGridData.grids[currentFloor][row].length; col++) {
            originalGridData.grids[currentFloor][row][col] = 'empty';
            updates.push({ floor: currentFloor, row, col, type: 'empty' });
        }
    }
    await batchUpdateCells(updates);
    renderGrid(bufferedGridData.grids[currentFloor]);
}

async function addNewFloor() {
    const newFloor = originalGridData.grids[currentFloor].map(row => row.map(() => 'empty'));
    originalGridData.grids.push(newFloor);
    originalGridData.floors.push({
        elevation: originalGridData.floors[originalGridData.floors.length - 1].elevation + originalGridData.floors[originalGridData.floors.length - 1].height,
        height: originalGridData.floors[originalGridData.floors.length - 1].height
    });
    await applyWallBuffer();
    currentFloor = bufferedGridData.grids.length - 1;
    renderGrid(bufferedGridData.grids[currentFloor]);
    updateFloorDisplay();
}

async function removeCurrentFloor() {
    if (bufferedGridData.grids.length > 1) {
        originalGridData.grids.splice(currentFloor, 1);
        originalGridData.floors.splice(currentFloor, 1);
        await applyWallBuffer();
        currentFloor = Math.min(currentFloor, bufferedGridData.grids.length - 1);
        renderGrid(bufferedGridData.grids[currentFloor]);
        updateFloorDisplay();
    } else {
        showError('Cannot remove the last floor.');
    }
}

async function findPath() {
    if (!start || goals.length === 0) {
        showError('Please set start and at least one goal.');
        return;
    }

    try {
        const response = await fetch('/api/find-path', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                grids: bufferedGridData.grids,
                grid_size: bufferedGridData.grid_size,
                floors: bufferedGridData.floors,
                bbox: bufferedGridData.bbox,
                start: start,
                goals: goals,
                allow_diagonal: allowDiagonal,
                minimize_cost: minimizeCost
            })
        });
        const data = await response.json();
        if (response.ok) {
            pathData = data.path;
            displayPathResult(data.path_lengths);
            renderGrid(bufferedGridData.grids[currentFloor]);
        } else {
            showError(`Pathfinding error: ${data.error}`);
        }
    } catch (error) {
        console.error('Error:', error);
        showError(`An error occurred while finding the path: ${error.message}`);
    }
}

function displayPathResult(pathLengths) {
    let resultHTML = `<h3>Path Lengths:</h3>
                      <p>Total length: ${pathLengths.total_length.toFixed(2)} meters</p>
                      <p>Distance to stairway: ${pathLengths.stairway_distance.toFixed(2)} meters</p>
                      <h4>Floor-wise lengths:</h4>
                      <ul>`;
    
    for (const [floor, length] of Object.entries(pathLengths.floor_lengths)) {
        resultHTML += `<li>Floor ${parseInt(floor.split('_')[1]) + 1}: ${length.toFixed(2)} meters</li>`;
    }
    
    resultHTML += '</ul>';
    
    document.getElementById('result').innerHTML = resultHTML;
}

function highlightPath(path) {
    renderGrid(bufferedGridData.grids[currentFloor]);
    const canvas = gridContainer.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(255, 255, 0, 0.5)';
    path.forEach(point => {
        if (point[2] === currentFloor) {
            ctx.fillRect(point[1] * cellSize, point[0] * cellSize, cellSize, cellSize);
        }
    });

    // Highlight start and goals
    ctx.fillStyle = 'green';
    if (start.floor === currentFloor) {
        ctx.beginPath();
        ctx.arc((start.col + 0.5) * cellSize, (start.row + 0.5) * cellSize, cellSize / 2, 0, 2 * Math.PI);
        ctx.fill();
    }

    ctx.fillStyle = 'red';
    goals.forEach(goal => {
        if (goal.floor === currentFloor) {
            ctx.beginPath();
            ctx.arc((goal.col + 0.5) * cellSize, (goal.row + 0.5) * cellSize, cellSize / 2, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
}

async function calculateEscapeRoutes() {
    if (!spacesData || spacesData.length === 0 || !goals || goals.length === 0) {
        showError('Please generate spaces and set exits before calculating escape routes.');
        return;
    }

    try {
        const response = await fetch('/api/calculate-escape-routes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                grids: bufferedGridData.grids,
                grid_size: bufferedGridData.grid_size,
                floors: bufferedGridData.floors,
                bbox: bufferedGridData.bbox,
                spaces: spacesData,
                exits: goals.map(goal => [goal.row, goal.col, goal.floor]),
                allow_diagonal: allowDiagonal
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log(data.escape_routes);
        foundEscapeRoutes = Object.values(data.escape_routes);
        renderGrid(bufferedGridData.grids[currentFloor]);
    
        // Display path lengths
        let totalLength = 0;
        let stairwayDistance = 0;
        let spacesOverMaxDistance = [];
        foundEscapeRoutes.forEach(route => {
            totalLength = Math.max(totalLength, route.distance);
            stairwayDistance = Math.max(stairwayDistance, route.distance_to_stair || 0);
            if (route.distance_to_stair > maxStairDistance) {
                spacesOverMaxDistance.push(route.id);
            }
        });
    
        const pathLengthsElement = document.getElementById('path-lengths');
        pathLengthsElement.innerHTML = `
            <h3>Escape Route Lengths:</h3>
            <p>Longest escape route: ${totalLength.toFixed(2)} meters</p>
            <p>Longest distance to stairway: ${stairwayDistance.toFixed(2)} meters</p>
            <p>Spaces over max stair distance: ${spacesOverMaxDistance.join(', ') || 'None'}</p>
        `;
    
    } catch (error) {
        console.error('Error:', error);
        showError(`An error occurred while calculating escape routes: ${error.message}`);
    }
}

function handleMaxStairDistanceChange(e) {
    maxStairDistance = parseInt(e.target.value);
    maxStairDistanceDisplay.textContent = `${maxStairDistance} m`;
    if (foundEscapeRoutes) {
        renderGrid(bufferedGridData.grids[currentFloor]);
    }
}

function generateRandomColor() {
    const hue = Math.floor(Math.random() * 360);
    return `hsl(${hue}, 70%, 80%)`;
}

function updateBufferForPaintedCells() {
    // This function should now use applyWallBuffer
    applyWallBuffer();
}

async function applyWallBuffer() {
    if (!Array.isArray(originalGridData.grids) || !originalGridData.grids.every(Array.isArray)) {
        console.error('Invalid grid data structure', originalGridData.grids);
        showError('Invalid grid data structure. Please refresh the page and try again.');
        return;
    }

    try {
        const response = await fetch('/api/apply-wall-buffer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                grids: originalGridData.grids,
                wall_buffer: wallBuffer,
                grid_size: originalGridData.grid_size,
                floors: originalGridData.floors,
                bbox: originalGridData.bbox
            })
        });
        const data = await response.json();
        if (response.ok) {
            if (Array.isArray(data.buffered_grids) && data.buffered_grids.every(Array.isArray)) {
                bufferedGridData = {...originalGridData, grids: data.buffered_grids};
                renderGrid(bufferedGridData.grids[currentFloor]);
            } else {
                console.error('Received invalid grid data from server', data.buffered_grids);
                showError('Received invalid data from server. Please try again.');
            }
        } else {
            throw new Error(data.error || 'An error occurred while applying the wall buffer.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred while applying the wall buffer.');
    }
}

async function updateCell(floor, row, col, cellType) {
    try {
        const response = await fetch('/api/update-cell', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                grids: originalGridData.grids,
                floor: floor,
                row: row,
                col: col,
                cell_type: cellType,
                wall_buffer: wallBuffer,
                grid_size: originalGridData.grid_size,
                floors: originalGridData.floors,
                bbox: originalGridData.bbox
            })
        });
        const data = await response.json();
        if (response.ok) {
            originalGridData.grids = data.original_grids;
            bufferedGridData = {...originalGridData, grids: data.buffered_grids};
        } else {
            throw new Error(data.error || 'An error occurred while updating the cell.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred while updating the cell.');
    }
}

async function batchUpdateCells(updates) {
    try {
        const response = await fetch('/api/batch-update-cells', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                grids: originalGridData.grids,
                updates: updates,
                wall_buffer: wallBuffer,
                grid_size: originalGridData.grid_size,
                floors: originalGridData.floors,
                bbox: originalGridData.bbox
            })
        });
        const data = await response.json();
        if (response.ok) {
            originalGridData.grids = data.original_grids;
            bufferedGridData = {...originalGridData, grids: data.buffered_grids};
        } else {
            throw new Error(data.error || 'An error occurred while updating cells.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred while updating cells.');
    }
}

async function detectExits() {
    try {
        const response = await fetch('/api/detect-exits', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                grids: bufferedGridData.grids,
                grid_size: bufferedGridData.grid_size,
                floors: bufferedGridData.floors,
                bbox: bufferedGridData.bbox
            })
        });
        const data = await response.json();
        if (response.ok) {
            goals = data.exits.map(exit => ({ row: exit[0], col: exit[1], floor: exit[2] }));
            renderGrid(bufferedGridData.grids[currentFloor]);
            //showMessage('Exits detected and added as goals.');
        } else {
            showError(`Exit detection error: ${data.error}`);
        }
    } catch (error) {
        console.error('Error:', error);
        showError(`An error occurred while detecting exits: ${error.message}`);
    }
}

function handleZoomChange(e) {
    zoomLevel = parseInt(e.target.value);
    cellSize = (zoomLevel / 100) * bufferedGridData.grid_size;
    updateZoomLevel();
    renderGrid(bufferedGridData.grids[currentFloor]);
}

function updateZoomLevel() {
    document.getElementById('zoom-level').textContent = `${zoomLevel/100}%`;
}

function handleBrushSizeChange(e) {
    brushSize = parseInt(e.target.value);
    document.getElementById('brush-size-display').textContent = brushSize;
}

function handleWallBufferChange(e) {
    wallBuffer = parseInt(e.target.value);
    const actualBufferSize = wallBuffer * bufferedGridData.grid_size;
    document.getElementById('wall-buffer-display').textContent = actualBufferSize.toFixed(2);
    applyWallBuffer();
}

function updateFloorDisplay() {
    document.getElementById('current-floor').textContent = `Floor: ${currentFloor + 1} / ${bufferedGridData.grids.length}`;
}

function showGridEditor() {
    document.getElementById('grid-editor').classList.remove('hidden');
    document.getElementById('pathfinder').classList.remove('hidden');
}

function showProgress(message) {
    progressContainer.classList.remove('hidden');
    progressBar.style.width = '0%';
    progressText.textContent = message;
}

function updateProgress(percentage, message) {
    progressBar.style.width = `${percentage}%`;
    progressText.textContent = `${percentage.toFixed(1)}%: ${message}`;
}

function hideProgress() {
    progressContainer.classList.add('hidden');
}

function showError(message) {
    console.error(message);
    alert(message);  // You can replace this with a more sophisticated error display method
}

function showMessage(message) {
    console.log(message);
    alert(message);  // You can replace this with a more sophisticated message display method
}

function downloadGrid() {
    if (!originalGridData) {
        showError('No grid data available. Please upload or create a grid first.');
        return;
    }

    const dataToSave = {
        grids: originalGridData.grids,
        grid_size: originalGridData.grid_size,
        floors: originalGridData.floors,
        bbox: originalGridData.bbox
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

function exportPath() {
    if (!pathData || pathData.length === 0) {
        showError('No path to export. Please find a path first.');
        return;
    }

    const pathByFloor = {};
    pathData.forEach(point => {
        const [x, y, floor] = point;
        if (!pathByFloor[floor]) {
            pathByFloor[floor] = [];
        }
        pathByFloor[floor].push([x, y]);
    });

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const floorCount = Object.keys(pathByFloor).length;
    const padding = 20;
    const floorHeight = (bufferedGridData.grids[0].length * cellSize) + padding;
    
    canvas.width = bufferedGridData.grids[0][0].length * cellSize;
    canvas.height = floorHeight * floorCount;

    Object.entries(pathByFloor).forEach(([floor, path], index) => {
        const yOffset = index * floorHeight;
        
        // Draw floor label
        ctx.fillStyle = 'black';
        ctx.font = '16px Arial';
        const pathLengths = JSON.parse(document.getElementById('result').getAttribute('data-path-lengths'));
        let yPosition = canvas.height - padding;
        ctx.fillText(`Total path length: ${pathLengths.total_length.toFixed(2)} meters`, padding, yPosition);
        yPosition -= 20;
        ctx.fillText(`Distance to stairway: ${pathLengths.stairway_distance.toFixed(2)} meters`, padding, yPosition);
        for (const [floor, length] of Object.entries(pathLengths.floor_lengths)) {
            yPosition -= 20;
            ctx.fillText(`Floor ${parseInt(floor.split('_')[1]) + 1} length: ${length.toFixed(2)} meters`, padding, yPosition);
        }    

        // Draw grid
        bufferedGridData.grids[floor].forEach((row, i) => {
            row.forEach((cell, j) => {
                ctx.fillStyle = getCellColor(cell);
                ctx.fillRect(j * cellSize, yOffset + i * cellSize + padding, cellSize, cellSize);
            });
        });

        // Draw path
        ctx.strokeStyle = 'blue';
        ctx.lineWidth = 2;
        ctx.beginPath();
        path.forEach(([x, y], i) => {
            const canvasX = y * cellSize + cellSize / 2;
            const canvasY = yOffset + x * cellSize + cellSize / 2 + padding;
            if (i === 0) {
                ctx.moveTo(canvasX, canvasY);
            } else {
                ctx.lineTo(canvasX, canvasY);
            }
        });
        ctx.stroke();

        // Draw start and end points
        if (index === 0) {
            ctx.fillStyle = 'green';
            const [startX, startY] = path[0];
            ctx.beginPath();
            ctx.arc(startY * cellSize + cellSize / 2, yOffset + startX * cellSize + cellSize / 2 + padding, cellSize / 2, 0, 2 * Math.PI);
            ctx.fill();
        }
        if (index === floorCount - 1) {
            ctx.fillStyle = 'red';
            const [endX, endY] = path[path.length - 1];
            ctx.beginPath();
            ctx.arc(endY * cellSize + cellSize / 2, yOffset + endX * cellSize + cellSize / 2 + padding, cellSize / 2, 0, 2 * Math.PI);
            ctx.fill();
        }
    });

    // Add path length at the bottom
    ctx.fillStyle = 'black';
    ctx.font = '16px Arial';
    ctx.fillText(`Path length: ${pathLength.toFixed(2)} meters`, padding, canvas.height - padding);

    // Convert canvas to blob and download
    canvas.toBlob(function(blob) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'path_export.png';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    });
}

// Initialize the application
function init() {
    // Set initial values for sliders
    zoomSlider.min = 100;  // 10% minimum zoom
    zoomSlider.max = 25000; // 2000% maximum zoom
    zoomSlider.value = zoomLevel;
    brushSizeSlider.value = brushSize;
    wallBufferSlider.value = wallBuffer;

    // Update displays
    document.getElementById('brush-size-display').textContent = brushSize;
    const initialBufferSize = wallBuffer * bufferedGridData.grid_size;
    document.getElementById('wall-buffer-display').textContent = initialBufferSize.toFixed(2);

}

// Call the init function when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', init);