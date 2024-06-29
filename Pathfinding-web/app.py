from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import json
from ifc_processor import create_navigation_grid
from grid_editor import InteractiveGridEditor
from pathfinder import InteractiveBIMPathfinder

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'ifc', 'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process-file', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.lower().endswith('.ifc'):
                grid_size = float(request.form.get('grid_size', 0.2))
                grids, bbox, floors = create_navigation_grid(filepath, grid_size=grid_size)
            elif filename.lower().endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                grids = data['grids']
                bbox = data['bbox']
                floors = data['floors']
                grid_size = data['grid_size']

            return jsonify({
                'grids': grids if isinstance(grids[0], list) else [grid.tolist() for grid in grids],
                'bbox': bbox,
                'floors': floors,
                'grid_size': grid_size
            })
        except Exception as e:
            return jsonify({'error': f'An error occurred while processing the file: {str(e)}'})
    return jsonify({'error': 'Invalid file'})


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