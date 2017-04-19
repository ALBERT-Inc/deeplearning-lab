import re
import requests
from requests.exceptions import RequestException, Timeout
from bs4 import BeautifulSoup

BASE_URL = "http://www.nissen.co.jp"

START_URL = "http://www.nissen.co.jp/nsrch/search.aspx?page=1&shop=0&category_main_name=&from=1&sort=new&ipg=60&AllColor=0&sf=4"
MENS_FASHION = "http://www.nissen.co.jp/cate011/sho_index/cate011_140_000_000-01.htm"
MENS_TOPS = "http://www.nissen.co.jp/cate006/sho_index/cate006_002_000_000-01.htm"

## ヘルパー関数（一般）
def get_soup(url):
    print("Try to get:", url)
    r = requests.get(url, timeout=1)
    r.encoding = "shift_jis" #ニッセンの場合
    soup = BeautifulSoup(r.text, "html.parser")
    return soup


## ヘルパー関数（一覧ページ）
def get_next_url(soup):
    next_page = soup.find("a", string="次へ")
    if next_page:
        next_url = BASE_URL + next_page.get("href")
        return next_url
    else:
        return None


def iter_over_pages(url0):
    next_url = url0
    while next_url:
        try:
            soup = get_soup(next_url)
        except TimeoutError:
            print('Timeout!')
            break
        
        yield soup
        next_url = get_next_url(soup)


def get_items(soup):
    items = soup.find_all('p', {'class': 'thumb'})
    item_urls = [BASE_URL + item.a['href'] for item in items]
    return item_urls

    
def get_all_items(url0):
    all_pages = iter_over_pages(url0)

    all_items = []
    for i, soup in enumerate(all_pages, start=1):
        print("on page", i, ":", soup.title.string)
        all_items += get_items(soup)
        
    return all_items


def get_item_images(soup):
    def complete_url(s):
        if "http" not in s:
            return BASE_URL + s
        else:
            return s

    def get_fname(url):
        return url.split("/")[-1]

    img_tags = soup.find_all("img")
    targets = [complete_url(t.get("target")) for t in img_tags if t.has_attr("target")]
    return {get_fname(url) : url for url in targets}


## スクレイピング
def scrape(url, download_imgs=True):
    """アイテムのページを受けとりスクレイピング結果を返す。画像をダウンロードする"""

    import re

    soup = get_soup(url)

    # "www.nissen.co.jp/.../<ID>.asp" という形のURLを期待        
    item_id = url.split('/')[-1][:10]

    # スクレイピング
    def tag_to_str(tag):
        if tag:
            return ''.join(tag.strings)
        else:
            return None

    def get_prices(soup):
        price_table = soup.find('table', attrs={'class': 'price'})
        if price_table:
            price_to_int = lambda s: int(re.sub('[^0-9]', '', s))
            prices = [price_to_int(s) for s in price_table.strings if '￥' in s]
        else:
            prices = []
        return prices

    breadcrumb = soup.find('div', attrs={'id': 'topicPath'})
    if breadcrumb:
        subcategories = [t.string for t in breadcrumb.find_all('a') if t.string != "カタログ通販 ニッセン TOP"]
    else:
        subcategories = []

    prices = get_prices(soup)
    description = tag_to_str(soup.find('p', itemprop='description'))

    
    details = tag_to_str(soup.find('div', id='DetailArea'))
    details = details.replace('\n商品について\n', '')
    details = details.replace(description, '')
    details = details.replace('上に戻る', '')
    details = re.sub('\n+', '\n', details)

    record = {"item_id" : item_id,
              "subcategories" : subcategories,
              "prices" : prices,
              "description" : description,
              "details" : details}

    print(record)
    img_urls = get_item_images(soup)

    return record, img_urls


def save_imgs(urls):
    print("Saving images...")
    for name, src in urls.items():
        r = requests.get(src, timeout=1)
        with open("./images/" + name + '.jpg', 'wb') as f:
            f.write(r.content)
    print("Done.")


if __name__ == '__main__':
    from itertools import repeat
    import json
    from time import sleep
    import os
    
    all_item_urls = get_all_items(MENS_TOPS)


    if not os.path.exists("./images"):
        os.mkdir("./images")

    # 初期化
    status = {l : False for l in all_item_urls}
    table = []

    # メインループ
    for i in range(3):
        print("Pass", i + 1)
        for url in status:
            if status[url]: # 既に成功していたら飛ばす
                continue
            else:
                sleep(1)

            try:
                record, img_urls = scrape(url)
                save_imgs(img_urls)
                table.append(record)
                status[url] = True
            except Timeout:
                print("Timeout! Skip it.")
            except RequestException:
                print("Failed! Skip it.")

        sleep(5)

    with open("data.json", "wt") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with open("log.json", "wt") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
        f.write("\n")
