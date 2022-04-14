import sqlite3
import re
import requests
from bs4 import BeautifulSoup


db = sqlite3.connect('VnExpress.db')
cursor = db.cursor()

url = "https://vnexpress.net/"
all_contents = ''
def get_db_name(url):
    """Lấy ra tên bảng (VnExpress) đổng thời in hoa chữ cái đầu"""
    url_clense = re.findall('ht.*://(.*?)\.',url)  #in hoa chữ cái đầu tiên là chữ V. từ vnexpress --> Vnxpress
    url_clense = url_clense[0].capitalize()
    return url_clense


#Lấy tên bảng Vnexpress
db_name = get_db_name(url)

# Tạo database, nếu database chưa tồn tại thỉ tiến hành tạo bảng mới
cursor.execute("CREATE TABLE IF NOT EXISTS " + db_name + " (STT"
" INTEGER PRIMARY KEY AUTOINCREMENT,URL varchar(255),Title varchar(255)"
",Description varchar(255), PageContents TEXT, Categories varchar(255))")

#Tạo list chứaa tất cả url
all_urls = []
all_urls.append("https://vnexpress.net/khao-sat-giao-vien-bang-bai-thi-cua-hoc-sinh-4254662.html")

def replace_token(text):
    text = text.replace('\t', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    return  text

def extract_content(soup):
    """Lấy ra các trường dữ liệu cần crawl"""
    title = soup.find("h1", class_="title-detail").text                                         #lấy ra tiêu đề
    description = soup.find("div", class_="sidebar-1").find("p").text               #lấy ra trích dẫn
    canonical = soup.find("ul", class_="breadcrumb").find('a').text             #lấy ra thể loại
    all_contents = soup.find('article', class_="fck_detail").text               #lấy ra nội dung
    title = replace_token(title)
    description = replace_token(description)
    canonical = replace_token(canonical)
    all_contents = replace_token(all_contents)
    return (title, description, all_contents, canonical)


def extract_links(soup):
    """"Lấy toàn bộ link của website"""
    links_dirty = soup.find_all('a') #tìm kiếm tất cả các thẻ a thể lấy thuộc tính [href]"
    for link in links_dirty:
        #lấy link trong từ th[uộc tính href, với điều kiện những link này phải bắt đầu bằng url (http://vnexpress.) và không được trùng với những link hợp lệ (với điều kiện 1) trước đó
        if str(link.get('href')).startswith(url) == True and link.get('href') not in all_urls:
            #nếu trong đường link có chứa các loại đuôi .jpg  .png  .#box_comment_vne thì bỏ qua
            if '.jpg' in link.get('href') or '.png' in link.get('href') or '#box_comment_vne' in link.get('href'):
                continue
            else:
                #thêm link vào list
                all_urls.append(link.get('href'))
    #trả về số lượng link ban đầu đã lấy được
    return (len(links_dirty))



def insert_data(extracted_data):
    """Chèn dữ liệu đã crawl được vào trong database"""
    url,title, description, contents, canonical = extracted_data
    cursor.execute("INSERT INTO " + db_name + " (URL, Title, "
    "Description, PageContents, Categories) VALUES(?,?,?,?,?)",
    (url, title, description, contents, canonical))
    db.commit()


link_counter = 0
#chạy vòng lặp để duyệt tất cả các link
while link_counter < len(all_urls):
    try:

        #yêu cầu truy cập vào các link, gửi một request đến từng link
        r = requests.get(all_urls[link_counter])
        if r.status_code == 200: #200 này là mã trạng thái HTTP, nó có nghĩa là "OK" (EG: Máy chủ đã trả lời thành công yêu cầu http).
            #html = r.text
            respones = requests.get(all_urls[link_counter]).content #lấy toàn bộ nội dung từ trang web
            soup = BeautifulSoup(respones, "html.parser") #format lại nội dung theo định dạng html
            extract_links(soup) #tìm kiếm tất cả những đường link có trong trang web vnexpress
            title, description, contents, canonical = extract_content(soup) #lấy ra các nội dung cần crawl có trong html vừa chuyển đổi

            if (canonical=='Ý kiến') :
                print(str(link_counter) + " crawling: " + all_urls[link_counter])
                print(canonical)
                insert_data((all_urls[link_counter], title, description, contents, canonical)) #chèn dữ liệu vào database
        link_counter += 1
    except Exception as e:
        link_counter += 1
        print(str(e))

cursor.close()
db.close()




















#1824801040107
#1824801040112