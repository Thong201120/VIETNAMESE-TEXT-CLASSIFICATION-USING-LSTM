import sqlite3

from flask import Flask, render_template, request, url_for
from flask import request
import utils as utils
import string
import pickle
import subprocess
import numpy as np
import datetime
import pandas as pd
from flask import json
from keras_preprocessing.text import tokenizer_from_json
from pyvi.ViTokenizer import ViTokenizer
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('TrangChu.html')

@app.route('/Introduction')
def Introduction():
    return render_template("Introduction.html")

@app.route('/Predict', methods = ['GET', 'POST'])
@app.route('/Predict', methods = ['GET', 'POST'])
def SEARCHdata():
    if request.method == 'GET':
        return render_template("PredictText.html")
    else:
        text = request.form['text']
        MAX_SEQUENCE_LENGTH = 250
        conn = sqlite3.connect("VnExpress.db")
        col = conn.execute("SELECT * FROM Vnexpress").fetchall()
        labels = ['Du lịch', 'Giáo dục', 'Giải trí', 'Khoa học',
                  'Kinh doanh', 'Pháp luật', 'Số hóa', 'Sức khỏe',
                  'Thế giới', 'Thể thao', 'Thời sự', 'Tâm sự', 'Xe',
                  'Ý kiến', 'Đời sống']
        json_file = open('model.json', 'r')
        model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights("model.h5")
        text_lower = text.lower()  # chuyển dữ liệu từ viết hoa sang viết thường
        strg = ViTokenizer.tokenize(text_lower)
        text_token = xulytudung(strg)
        with open('tokenizer.json') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)

        seq = tokenizer.texts_to_sequences([text_token])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(padded)
        result = labels[np.argmax(pred)]
        return render_template("PredictText.html", text=text, result=result)

@app.route('/FilePredict', methods = ['GET', 'POST'])
@app.route('/FilePredict', methods = ['GET', 'POST'])
def Readfile():
    if request.method == 'GET':
        return render_template("UploadFile.html")
    else:
        f = request.files['file']
        text = f.read()
        text = text.decode('utf-8')

        MAX_SEQUENCE_LENGTH = 250
        conn = sqlite3.connect("VnExpress.db")
        col = conn.execute("SELECT * FROM Vnexpress").fetchall()
        labels = ['Du lịch', 'Giáo dục', 'Giải trí', 'Khoa học', 'Kinh doanh', 'Pháp luật', 'Số hóa', 'Sức khỏe',
                  'Thế giới', 'Thể thao', 'Thời sự', 'Tâm sự', 'Xe', 'Ý kiến', 'Đời sống']
        json_file = open('model.json', 'r')
        model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights("model.h5")

        text_lower = text.lower()  # chuyển dữ liệu từ viết hoa sang viết thường
        strg = ViTokenizer.tokenize(text_lower)
        text_token = xulytudung(strg)

        with open('tokenizer.json') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)

        seq = tokenizer.texts_to_sequences([text_token])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(padded)
        result = labels[np.argmax(pred)]
        return render_template("PredictFile.html", text=text, result=result)

@app.route("/searchdata")
def GET_RESULTS ():
    return render_template("Data.html")

@app.route('/Data', defaults={'page': 1}, methods = ['GET', 'POST'])
@app.route('/Data<int:page>', methods = ['GET', 'POST'])
def DATA(page):
    rows = utils.get_all("SELECT * FROM VnExpress")
    data = []
    index = 1
    for row in rows[page*50-50: page*50]:
        data.append(
            {
                "STT": index,
                "id": row[0],
                "theloai": row[5],
                "title": row[2],
                "sumary": row[3],
                "content": row[4]
            }

        )
        index += 1

    return render_template("data.html", data=data, id = page)

@app.route('/chitiet_bao')
def CHITIET_BAO():
    rows = utils.get_all("SELECT * FROM CT")
    data = []
    for row in rows:
        data.append(
            {
                "id": row[0],
                "theloai": row[1],
                "title": row[2],
                "sumary": row[3],
                "TEXT": row[4]
            }
        )
    return render_template("ChiTiet.html", data=data)

@app.route('/JSchart')
def CHITIEAO():
    return render_template("Statistic.html")

@app.route("/news/<int:news_id>", methods = ["GET"])
def get_news_by_id(news_id):
    row = utils.get_news_id(news_id)
    data = {
        "id": row[0],
        "theloai": row[5],
        "title": row[2],
        "sumary": row[3],
        "content": row[4]
    }
    return render_template("Detail.html", data=data)



@app.route('/shutdown', methods=['POST'])

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

def shutdown():
    shutdown_server()
    return 'Server shutting down...'


def xulytudung(comment):
    stop_word = [] #tạo list để chứa stopword
    #mở file stopword
    with open("vietnamese-stopwords-dash.txt", encoding="utf-8") as f:
        text = f.read()
        for word in text.split():
            stop_word.append(word) #thêm từng stopword vào list
        f.close()
    remove = string.punctuation #string.punctuation chứa các ký tự !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    remove = remove.replace("|", "")  # không loại bỏ ký tự |, kí tự dùng để phân biệt tiêu đề, trích dẫn, nội dung
    punc = list(remove)
    stop_word = stop_word + punc #list top word lúc này có chứa các ký tự đặc biệt
    str = '' #tạo một chuổi để lưu lại text đã loại stopword
    for word in comment.split(" "): #tách câu thành từng từ dựa vào khoảng trắng
        if (word not in stop_word):
            if ("_" in word) or ("|" in word) or (word.isalpha() == True): #nếu là dấu gạch dưới, dấu gạch đứng, chữ cái hay số thì sẽ cộng lại thành chuổi
                #không bỏ dấu gạch dưới vì dấu này đang dùng để liên kết cụm từ với nhau
                #dấu gạch đứng để phân biệt nội dung trong câu
                str = str + word + " "
    return str


# set FLASK_APP="tenfile"
if __name__ == '__main__':
    app.run(debug = True)