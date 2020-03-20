#request 获取网页源码
#beautifulsoup对内容进行解析
#将解析好的内容存在excel文件里面，写入excel文件； xlrt 读取excel文件

from bs4 import BeautifulSoup as bs
import requests
import xlwt

workbook = xlwt.Workbook(encoding='utf-8')  # 创建一个workbook并且设置编码
worksheet = workbook.add_sheet('豆瓣电影 Top 100')  # 创建一个表格，命名为“豆瓣电影 Top 100”
worksheet.write(0, 0, '电影排名')
worksheet.write(0, 1, '中文名称')
worksheet.write(0, 2, '别名')
worksheet.write(0, 3, '导演&主演')
worksheet.write(0, 4, '国家/地区')
worksheet.write(0, 5, '上映时间')
worksheet.write(0, 6, '类型')
worksheet.write(0, 7, '评分')
worksheet.write(0, 8, '引述')
row = 1


def main(number):
    url = 'https://movie.douban.com/top250?start='+str(number)+'&filter='
    #print('url:', url)
    html = request_douban_data(url)
    #print('html:', html)
    soup = bs(html, 'html.parser')
    lists = soup.find('ol', class_='grid_view').find_all('li')
    #print('lists:', lists)

    global row
    for each in lists:
        #print(each)
        each_movie_index = each.find(class_='pic').find('em').string  # 电影排名
        each_movie_name = each.find(class_='info').find(class_='hd').find(class_='title').string  # 中文名称
        each_movie_name1 = each.find_all('span')[1].get_text().split('/')[1]  # 别名
        each_movie_role = each.p.next_element.replace("\n", "").replace(" ", "")  # 导演&主演
        each_movie_region = each.p.get_text().split('/')[-2]  # 国家/地区
        each_movie_time = each.p.get_text().split('\n')[2].split('/')[0].replace(" ", "")  # 上映时间
        each_movie_type = each.p.get_text().split('\n')[2].split('/')[-1]  # 类型
        each_movie_details = each.find(class_='info').find(class_='bd').find(class_='').string  # 电影详细信息
        each_movie_star = each.find(class_='star').find(class_='rating_num').text  # 评分
        each_movie_quote = each.find(class_='quote').find('span').string  # 引述

        print('电影排名：'+each_movie_index+'|'+'中文名称：'+each_movie_name+'|'+'别名：'+each_movie_name1+'|'+each_movie_role+'|'+'国家/地区：'+each_movie_region+'|'+'上映时间：'+each_movie_time+'|'+'类型：'+each_movie_type+'|'+'评分：' + each_movie_star + '|' + '引述：' + each_movie_quote)
        print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        worksheet.write(row, 0, each_movie_index)
        worksheet.write(row, 1, each_movie_name)
        worksheet.write(row, 2, each_movie_name1)
        worksheet.write(row, 3, each_movie_role)
        worksheet.write(row, 4, each_movie_region)
        worksheet.write(row, 5, each_movie_time)
        worksheet.write(row, 6, each_movie_type)
        worksheet.write(row, 7, each_movie_star)
        worksheet.write(row, 8, each_movie_quote)

        row = row + 1


def request_douban_data(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.18362'}
        response = requests.get(url, headers=headers)
        #print('response.status_code:', response.status_code)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        return None


if __name__ == '__main__':
    i = 0
    for i in range(4):
        number = i*25
        main(number)
    workbook.save('../movie list.xls')
