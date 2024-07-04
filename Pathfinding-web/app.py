import subprocess
import sys
import threading

import ifcopenshell
from flask import Flask, render_template, request, jsonify, Response, session
from werkzeug.utils import secure_filename
import os
import json
from ifc_processor import create_navigation_grid, calculate_bounding_box_and_floors, create_faux_3d_grid, all_types, \
    process_element, trim_and_pad_grids
from grid_editor import InteractiveGridEditor
import pathfinder
from pathfinder import InteractiveBIMPathfinder
import numpy as np

app = Flask(__name__)
app.secret_key = '1234'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'ifc', 'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def expand_mask(mask):
    expanded = mask.copy()
    rows, cols = mask.shape
    for i in range(rows):
        for j in range(cols):
            if mask[i, j]:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if 0 <= i + di < rows and 0 <= j + dj < cols:
                            expanded[i + di, j + dj] = True
    return expanded

@app.route('/apply-wall-buffer', methods=['POST'])
def apply_wall_buffer_route():
    data = request.json
    pathfinder.grids = data['grids']
    pathfinder.wall_buffer = data['wall_buffer']
    buffered_grids = pathfinder.apply_wall_buffer()
    return jsonify({'buffered_grids': [grid.tolist() for grid in buffered_grids]})

@app.route('/update-buffer', methods=['POST'])
def update_buffer():
    data = request.json
    global pathfinder
    pathfinder = InteractiveBIMPathfinder(data['grids'], data['grid_size'], data['floors'], data['bbox'])
    pathfinder.grids = data['grids']
    pathfinder.wall_buffer = data['wall_buffer']
    buffered_grids = pathfinder.apply_wall_buffer()
    updated_floor = buffered_grids[data['floor']]#pathfinder.update_buffer_for_cells(data['floor'], data['affected_cells'], data['wall_buffer'])
    return jsonify({'updated_floor': updated_floor.tolist()})

def apply_wall_buffer(grid, buffer_distance):
    buffered_grid = grid.copy()
    wall_mask = (grid == 'wall')

    for _ in range(buffer_distance):
        wall_mask = expand_mask(wall_mask)

    rows, cols = wall_mask.shape
    for i in range(rows):
        for j in range(cols):
            if wall_mask[i, j] and grid[i, j] not in ['wall', 'door', 'stair']:
                buffered_grid[i, j] = 'walla'

    return buffered_grid


@app.route('/')
def index():
    return render_template('index.html')


def process_ifc(file_path, grid_size):
    def event_stream():
        print("init " + os.path.dirname(os.path.realpath(__file__)))

        ifc_processor_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ifc_processor.py")
        absolute_file_path = os.path.abspath(file_path)
        print(f"IFC processor path: {ifc_processor_path}")
        print(f"File path: {absolute_file_path}")
        print(f"Grid size: {grid_size}")

        try:
            process = subprocess.Popen([sys.executable, ifc_processor_path, absolute_file_path, str(grid_size)],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       universal_newlines=True,
                                       cwd=os.path.dirname(os.path.realpath(__file__)))

            print("Subprocess started")

            # Read from stderr in a separate thread to prevent blocking
            def read_stderr():
                for line in process.stderr:
                    print(f"STDERR: {line.strip()}")

            stderr_thread = threading.Thread(target=read_stderr)
            stderr_thread.start()

            for line in process.stdout:
                print(f"STDOUT: {line.strip()}")
                if line.startswith("PROGRESS:"):
                    _, progress, message = line.strip().split(":", 2)
                    yield f"data: {json.dumps({'progress': float(progress), 'message': message})}\n\n"
                elif line.startswith("{") and line.strip().endswith("}"):
                    # This is likely the final JSON result
                    yield f"data: {json.dumps({'complete': True, 'result': json.loads(line)})}\n\n"

            print("Subprocess output finished")

            # Wait for the process to complete
            process.wait()
            print(f"Subprocess exit code: {process.returncode}")

            if process.returncode != 0:
                error_output = process.stderr.read()
                print(f"Error output: {error_output}")
                yield f"data: {json.dumps({'error': f'Processing failed: {error_output}'})}\n\n"

        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            yield f"data: {json.dumps({'error': f'An unexpected error occurred: {str(e)}'})}\n\n"
    
    return Response(event_stream(), content_type='text/event-stream')


@app.route('/process-file', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session['filepath'] = filepath
        session['grid_size'] = float(request.form.get('grid_size', 0.2))
        return jsonify({'message': 'File uploaded successfully'}), 200
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/process-file-sse')
def process_file_sse():
    filepath = session.get('filepath')
    grid_size = session.get('grid_size')
    if not filepath or not grid_size:
        return jsonify({'error': 'No file to process'}), 400
    return process_ifc(filepath, grid_size)


@app.route('/edit-grid', methods=['POST'])
def edit_grid():
    data = request.json
    editor = InteractiveGridEditor(data['grids'], data['grid_size'], data['floors'], data['bbox'])
    updated_grids = editor.edit_grid(data['edits'])
    return jsonify({'grids': updated_grids})


@app.route('/find-path', methods=['POST'])
def find_path():
    data = request.json
    pathfinder = InteractiveBIMPathfinder(data['grids'], data['grid_size'], data['floors'], data['bbox'])
    pathfinder.start = data['start']
    pathfinder.goals = data['goals']
    path = pathfinder.run_astar()
    return jsonify({'path': path})


if __name__ == '__main__':
    app.run(debug=True)