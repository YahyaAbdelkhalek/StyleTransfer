import os
import sys
import flask
from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime


app = Flask(__name__,static_folder='public')

public_folder = os.path.join(app.root_path, 'public')

if not os.path.exists(public_folder):
    os.makedirs(public_folder)


def save_image(file):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S") 
    filename = f"{timestamp}_{secure_filename(file.filename)}"
    file_path = os.path.join(public_folder, filename)
    file.save(file_path)
    return file_path

@app.route('/recognize', methods=['POST'])
def recognize():
    return send_from_directory(public_folder, '606px-Van_Gogh_-_Starry_Night_-_Google_Art_Project')
    #return jsonify({'error': 'No file part'})


if __name__ == '__main__':
    app.run(debug=True)
