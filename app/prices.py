from bs4 import BeautifulSoup
import requests

def get_prices(cpu_links):
    print(cpu_links["DNS"])
    test = parse_dns("https://www.dns-shop.ru/catalog/17a899cd16404e77/processory/?q=AMD+Ryzen+5+8400F&stock=now-today-tomorrow-later-out_of_stock&order=6")
    print("TEST == ", test)
    return test

def parse_dns(url):
    try:
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
        'Referer': 'https://www.dns-shop.ru/',
        }
        session = requests.Session()
        session.headers.update(headers)
        response = session.get(url)
        print(response.status_code)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        priceLink = {}
        for item in soup.select('.catalog-product'):
                link = item.select_one('catalog-product__name ui-link ui-link_black').get('href')
                price = item.select_one('.product-buy__price').text.strip()
                # priceLink.append({'link': link, 'price': price})
                print("link -- ", link)
                print("price -- ",price)
                priceLink['link'] = link
                priceLink['price'] = price
        
        return priceLink
    except Exception as e:
        print(f"Ошибка: {str(e)}")
    