import os
import time
from threading import Thread

import eventlet
from flask import Flask, send_from_directory, render_template, request, redirect

from config import ip_address, port
from ml import cache_articles, predict_document


# Init app
async_mode = None
app = Flask(__name__, static_url_path='')


# Return main page
@app.route('/')
def root():
    return render_template('index.html')


# Main process block of pdf file
@app.route('/process')
def process():
    arxiv_id = request.args.get('arxiv_id')
    print(arxiv_id)
    predict_document(arxiv_id, cache_folder, pdf_folder)

    return redirect(f"http://papers.pavtiger.com/pdf/{arxiv_id}/main.pdf", code=302)


# Get files from server (e.g libs)
@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


# We start a parallel thread for game logics. This loop is constantly running
def game_loop(name):
    while True:
        # Process game logic here if you need to
        time.sleep(0.01)


if __name__ == "__main__":
    # x = Thread(target=game_loop, args=(1,))
    # x.start()

    cache_folder = os.path.join("static", "cache")
    pdf_folder = os.path.join("static", "pdf")

    cache_articles()

    app.run(host=ip_address, port=port)

