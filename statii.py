from bs4 import BeautifulSoup
import requests

response = requests.get('https://arxiv.org/list/astro-ph.EP/current?show=100')

baba = BeautifulSoup(response.content, 'html.parser')

all_dds = baba.dl.find_all('dd')

for dd in all_dds:
