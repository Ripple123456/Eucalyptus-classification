import requests
import time
import pandas as pd
import os



######固定的
# api_key = 'AIzaSyBT1rRxNgqUM1fIlmksdRqP7jtEs80AMpk'
api_key= 'AIzaSyDghE0CewBgyoFoF0YBqbIuV_vKUwrZgnU'
engine_id = '8cadb5a5c9752e7ef'
# engine_id ='f06e49547b0fe4056'


# <script async src="https://cse.google.com/cse.js?cx=f06e49547b0fe4056">
# </script>
# <div class="gcse-search"></div>


#####要搜索的关键字
# search_term = 'Blue gum tree'
search_term= 'Eucalyptus cinerea'

total_pages = 20
num_per_page = 10
start_offset = 1
####图片链接
image_links = []

for page_num in range(start_offset, total_pages + 1):
    print(f"正在爬取第 {page_num} 页...")
    start_index = (page_num - 1) * num_per_page + 1
    url = f'https://www.googleapis.com/customsearch/v1?key={api_key}&cx={engine_id}&q={search_term}&start={start_index}&num={num_per_page}&searchType=image'
    try:
        response = requests.get(url,timeout=30).json()
        items = response['items']
        for item in items:

            link = item['link']
            print(link)
            image_links.append(link)
        time.sleep(1)
    except Exception as e:
        print(f"爬取第 {page_num} 页失败:{e}")
df = pd.DataFrame({
    '名称':[search_term+'-'+str(ii) for ii in range(1,len(image_links)+1)],
    '链接':image_links
})
df.to_csv('500谷歌图片搜索.csv',encoding='utf_8_sig',index=False)
print(f"总共采集到了 {len(image_links)} 张图片链接。")



print('-----------------------------------------------------现在开始下载图片------------------------------------------------------')

num = 0
dir_name = search_term
# 判断目录是否存在，不存在则创建目录
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

for url in image_links:
    try:
        r = requests.get(url)
        num += 1

        # Corrected file path using os.path.join
        file_path = os.path.join(dir_name, f"{search_term}-{num}.jpg")
        
        with open(file_path, 'wb') as file:
            file.write(r.content)
            print('写入完成：{}'.format(url))
    except Exception as e:
        print(f"下载失败：{url}, 错误信息：{e}")