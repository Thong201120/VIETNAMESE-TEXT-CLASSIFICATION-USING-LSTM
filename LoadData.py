import os
import sqlite3
import string
from pyvi import ViTokenizer

conn = sqlite3.connect("VnExpress.db")
# -----------------------------------------Test in dữ liệu và loại bỏ các ký tự thừa /n , /t ra khỏi nội dung
col = conn.execute("SELECT * FROM Vnexpress").fetchall()

#Tạo hai list rỗng để lưu tên tên loại và số lượng bài báo của từng thể loại
classify = conn.execute("SELECT Categories, COUNT(*) from %s group by Categories" % "Vnexpress")
categories = []
number_of_paper = []
for row in classify:
    categories.append(row[0])
    number_of_paper.append(row[1])
# print(number_of_paper)


# -------------------------------Tạo thư mục categories, dùng để chứa dữ liệu được format từ dưới csdl trên trên file text
root_path = "CATEGORIES\\"
for category in categories: #tạo các thư mục thể loại con để để dàng phân loại
   os.mkdir(os.path.join(root_path, category))

# -------------------------------Tạo thư mục removed, dùng để chứa các file text trong category đã được loại bỏ stopword và các ký tự không cần thiết
# root_path = "REMOVED\\"
# for category in categories:
#    os.mkdir(os.path.join(root_path, category))

#hàm xử lý từ dừng - stopwords
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

# ------------------------------------------Tiền Xử lí dữ liệu và lưu vào file text
for category in categories:
    analysis = conn.execute("SELECT * from Vnexpress where categories = '%s'" %str(category)) #lấy dữ liệu theo trường thể loại
    categories_data = []
    index = 1 #khai báo một biến index để tạo tên cho file txt
    #lấy dữ liệu từ file txt lên
    for row in analysis:
        data = []
        data.append(row[2])
        data.append(row[3])
        data.append(row[4])
        categories_data.append(data)
    #chèn dữ liệu vào từng file text theo từng thư mục thể loại đã được tạo trước đó
    for tungbaibao in categories_data:
        f = open(f"CATEGORIES/{str(category)}/{str(index)}.txt", 'w', encoding="utf8")
        #chèn ký hiệu "|" để phân biệt tiêu đề, trích dẫn cũng như nội dung
        text = str(tungbaibao[0]) + "|" + str(tungbaibao[1]) + "|" + str(tungbaibao[2])
        text_lower = text.lower()  # chuyển dữ liệu từ viết hoa sang viết thường
        strg = ViTokenizer.tokenize(text_lower)  # hàm tokenize sẽ bắt những cụm từ là tiếng việt có nghĩa và kết nối cụm từ dựa vào dấu gạch chân
        text_token = xulytudung(strg)  # sau khi phát hiện cụm từ sẽ tiến hành loại bỏ stopword việt nam
        text_token = text_token.split("|")  # cắt chuổi vừa xử lí dựa vào dấu gạch đứng và lưu vào file txt trong thư mục thể loại trong remove
        # f.write(str("__label__" + str(ViTokenizer.tokenize(category)).lower() + " " + text_token[0]) + " " + str(text_token[1]) + " " + str(text_token[2]))
        f.write(str(text_token[0]) + " " + str(text_token[1]) + " " + str(text_token[2]))

        index += 1
        f.close()
