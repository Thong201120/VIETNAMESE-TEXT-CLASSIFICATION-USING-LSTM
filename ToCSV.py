import sqlite3
import random
import pandas as pd

conn = sqlite3.connect("VnExpress.db")
col = conn.execute("SELECT * FROM Vnexpress").fetchall()

classify = conn.execute("SELECT Categories, COUNT(*) from %s group by Categories" % "Vnexpress")
categories = []
number_of_paper = []
for row in classify:
    categories.append(row[0])
    number_of_paper.append(row[1])

def train_text():
    count = 0
    all_text = []
    all_label = []
    for category in categories:
        analysis = conn.execute("SELECT * from Vnexpress where categories = '%s'" % str(category))
        index = 1
        for row in analysis:  # mở từng file txt đã xử lý stopword trong từng thể loại
            f = open(f"CATEGORIES/{str(category)}/{str(index)}.txt", 'r', encoding="utf8")
            text = f.read()
            all_text.append(text)
            all_label.append(category)
            index = index + 1
            f.close()
    print("all_text: ", end=' ')
    print(all_text[1:10])
    print("all_labels: ", end=' ')
    print(all_label)
    return all_text, all_label

def JoinList(Listcmt, Listpts):
    List_data = []
    for i in range(len(Listcmt)):
        each_list = []
        each_list.append(Listcmt[i])
        each_list.append(Listpts[i])
        List_data.append(each_list)
    return List_data

def SeperateList(List):
    res1, res2 = map(list, zip(*List))
    return res1, res2

def save_dataCSV():
    dir_data, dir_label = train_text()
    listData = JoinList(dir_data, dir_label)
    print(len(listData))
    list_Data = random.sample(listData, len(listData))


    alldata = pd.DataFrame(listData, columns=['data', 'label'])
    alldata.to_csv('data.csv')

save_dataCSV()