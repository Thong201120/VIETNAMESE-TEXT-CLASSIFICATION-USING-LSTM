import sqlite3
import string


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
from underthesea.transformer.tfidf import TfidfVectorizer



def get_all(query):
    conn = sqlite3.connect("VnExpress.db")
    data = conn.execute(query).fetchall()
    conn.close()
    return data

def get_news_id(news_id):
    conn = sqlite3.connect("VnExpress.db")
    query = """
        SELECT *
        FROM Vnexpress    
        WHERE STT=?
        """
    news = conn.execute(query,(news_id, )).fetchone()
    conn.close()
    return news
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
            if ("_" in word) or ("|" in word) or (word.isalpha() == True) or (word.isdigit() == True): #nếu là dấu gạch dưới, dấu gạch đứng, chữ cái hay số thì sẽ cộng lại thành chuổi
                #không bỏ dấu gạch dưới vì dấu này đang dùng để liên kết cụm từ với nhau
                #dấu gạch đứng để phân biệt nội dung trong câu
                str = str + word + " "
    return str

def manual_replace(s, char, index):
    return s[:index] + char + s[index +1:]

def search(tex):
    data = []
    all_news = []
    conn = sqlite3.connect("VnExpress.db")
    col = conn.execute("SELECT * FROM VnExpress").fetchall()

    classify = conn.execute("SELECT Categories, COUNT(*) from %s group by Categories" % "Vnexpress")
    categories = []
    number_of_paper = []
    for row in classify:
        categories.append(row[0])
        number_of_paper.append(row[1])

    for category in categories:
        analysis = conn.execute("SELECT * from Vnexpress where categories = '%s'" % str(category))
        searchtext = []
        index = 1
        for row in analysis:
            f = open(f"CATEGORIES/{str(category)}/{str(index)}.txt", 'r', encoding="utf8")
            text = f.read()
            text.replace("\n", " ")
            searchtext.append(text + "\n" + str(category))
            index += 1
            f.close()
        all_news = all_news + searchtext
    conn.close()
    vector = TfidfVectorizer() #tự động chuyển đổi một bộ dữ liệu thô sang dang ma trận và sửa dụng được các tính năng của tf-idf
    X = vector.fit_transform(all_news) #chuyển đổi dữ liệu sang ma trận để chuẩn bị tính toán
    # print(X)
    te = str(tex) #gán nội dung tìm kiếm
    te = te.lower() #chuyển nội dung tìm kiếm về dạng chữ viết thường
    te = word_tokenize(te, format='text') #kết nối các cụm từ có nghĩa nếu có xuất hiện trong từ khóa tìm kiếm
    te=xulytudung(te) #loại bỏ các stopword nếu có xuất hiện
    print(te)
    te = vector.transform([te]) #tính toán trọng số xuất hiện của từ nếu từ đó có xuất hiện trong toàn bộ text, nếu không có sẽ ko in gì cả
    print(te)
    m = str(te) # xử lí các kết quả trọng số
    # print(len(m))
    if len(m) > 1: #nếu chiều dài của str m  > 1 hay nói cách khác là có tồn tại ít nhất một trọng số (một từ khóa được tìm kiếm)
        re = cosine_similarity(X, te) #tính cosine giữa X và te là một ma trận có 2 cột và X dòng
        Li = []
        for i in range(len(re)):
            Li.append(re[i][0]) #thêm các giá trị của re vào list
        index = 1
        for i in np.argsort(Li)[-20:][::-1]: #sắp xếp các giá trị re trong e[elist Li theo thứ tự giảm dần về giá trị và lấy ra 50 giá trị cao nhất để biểu diễn kết quả tìm kiếm
            r = all_news[i].split("\n")
            r[1] = r[1].replace("_", " ")
            r[1] = manual_replace(r[1], '', 0)
            r[1] = r[1].capitalize()
            r[0] = r[0].replace("_", " ")
            r[0] = r[0].capitalize()
            r[2] = r[2].replace("_", " ")
            r[2] = manual_replace(r[2], '', 0)
            r[2] = r[2].capitalize()
            r[3] = r[3].replace("_", " ")
            r[3] = r[3].capitalize()
            r.append(index)
            index = index + 1
            data.append(r)

    else:
        print("Không tìm thấy kết quả cho từ khóa ")
    return data


if __name__ =="__main__":
    print(get_all("SELECT * FROM Vnexpress"))
