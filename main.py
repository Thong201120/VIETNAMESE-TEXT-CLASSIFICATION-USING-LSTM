# def JoinList(Listcmt, Listpts):
#     List_data = []
#     for i in range(len(Listcmt)):
#         each_list = []
#         each_list.append(Listcmt[i])
#         each_list.append(Listpts[i])
#         List_data.append(each_list)
#     return List_data
#
#
# data = ['fdsa', 'dsfadfdfaffadfd', 'sadfadfadfadfdfafasdfdsfd']
# label = ['kinh te', 'khoa hoc', 'xa hoi']
# print(JoinList(data, label))
import json
import sqlite3

from keras.models import model_from_json
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras_preprocessing.text import tokenizer_from_json
from pyvi.ViTokenizer import ViTokenizer
import string

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






MAX_SEQUENCE_LENGTH = 250
conn = sqlite3.connect("VnExpress.db")
col = conn.execute("SELECT * FROM Vnexpress").fetchall()
labels = ['Du lịch', 'Giáo dục', 'Giải trí', 'Khoa học', 'Kinh doanh', 'Pháp luật', 'Số hóa', 'Sức khỏe', 'Thế giới', 'Thể thao', 'Thời sự', 'Tâm sự', 'Xe', 'Ý kiến', 'Đời sống']
json_file = open('model.json', 'r')
model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("model.h5")
print(labels)


text = 'Luật sư cho rằng diễn viên Trịnh Sảng có thể đối diện án tù từ ba tới bảy năm nếu bị xác định trốn thuế.' \
       'Trên trang 163, Phong Dược Bình, luật sư nổi tiếng ở Bắc Kinh, phân tích mức phạt cho Trịnh Sảng phụ thuộc lớn vào việc cô sai phạm lần đầu hay ' \
       'tái phạm. Nếu là lần đầu trốn thuế, cô sẽ phải nộp phạt lớn hơn mức của Phạm Băng Băng năm 2018.' \
       'Ngày 29/4, công ty của Trịnh Sảng phản hồi: "Cơ quan thuế đã kiểm tra hợp đồng, thuế thu nhập, tất cả hợp đồng kinh tế của tôi. Tôi sẽ phối hợp điều ' \
       'tra và sẽ thông báo khi có kết quả. Cảm ơn các bạn quan tâm".' \
       'Nếu tái phạm, ngoài nộp phạt, Trịnh Sảng phải ngồi tù từ ba đến bảy năm. Theo Trương Hằng - người tố cáo Trịnh Sảng trốn thuế - nữ diễn viên ' \
       'tái phạm. Năm 2018, sau vụ Phạm Băng Băng, cô là một trong số hàng trăm sao gốc Hoa chủ động nộp bù thuế để tránh bị điều tra, xử phạt. Nếu Trịnh' \
       ' Sảng tiêu hủy chứng cứ hoặc không phối hợp với cơ quan chức năng, cô cũng sẽ bị xử lý hình sự.' \
       'Luật sư nhận định qua các ảnh chụp đoạn hội thoại mà Trương Hằng đăng trên Weibo, nhiều khả năng anh tham gia giúp Trịnh Sảng trốn thuế vì là' \
       ' quản lý của cô năm 2019. Trương Hằng và cha mẹ của Trịnh Sảng đều sẽ bị điều tra, làm rõ ai là chủ mưu, tòng phạm.' \
       'Phong Dược Bình sinh năm 1985, tốt nghiệp ngành Luật tại Đại học Glasgow (Anh). Anh là một trong người thành lập công ty luật Bejing Jingsh Law Firm.' \
       ' Anh còn giảng dạy, hướng dẫn nghiên cứu sinh thạc sĩ tại Học viện Quan hệ quốc tế (đại học công lập ở Bắc Kinh).' \
       'Hôm 26/4, Trương Hằng tố cáo Trịnh Sảng nhận thù lao 160 triệu nhân dân tệ (24,6 triệu USD) cho 77 ngày làm việc trên trường quay Thiện nữ u hồn, tuy nhiên trong hợp đồng, cát-xê của diễn viên là 48 triệu tệ (7,4 triệu USD). Số còn lại được chuyển cho công ty đứng tên cha mẹ Trịnh Sảng, như một khoản tăng vốn của công ty.' \
       'Theo Thepaper, sự việc của Trịnh Sảng có thể là lý do khiến làng giải trí biến động. Gần đây một loạt sao hủy giấy đăng ký kinh doanh công ty,' \
       ' gồm Ngụy Đại Huân, Đặng Siêu, Na Anh, Ngô Tuyên Nghi, Tỉnh Bách Nhiên, Vương Thiên Nguyên... Năm 2018, Phạm Băng Băng từng gây "cơn địa chấn" ' \
       'với showbiz khi bị điều tra sử dụng "hợp đồng âm dương" để trốn thuế. Cô bị phạt, truy thu thuế tổng số tiền 128 triệu USD. Sau đó, giới chức' \
       ' Trung Quốc cho phép các ngôi sao chủ động nộp bù thuế để tránh bị phạt như Băng Băng. Tính đến tháng 1/2019, nước này thu hơn 11 tỷ nhân dân ' \
       'tệ (gần 1,7 tỷ USD) tiền trốn thuế của người làm lĩnh vực giải trí.'


text_lower = text.lower()  # chuyển dữ liệu từ viết hoa sang viết thường
strg = ViTokenizer.tokenize(text_lower)
text_token = xulytudung(strg)

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

seq = tokenizer.texts_to_sequences([text_token])
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
print(pred, labels[np.argmax(pred)])