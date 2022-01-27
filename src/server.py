import numpy as np
from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from search import search_image
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        

        # Run search
        output_paths = search_image(img)

        return render_template('index.html',
                               paths = output_paths)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")