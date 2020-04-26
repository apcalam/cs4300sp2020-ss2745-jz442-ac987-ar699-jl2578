from bs4 import BeautifulSoup
import requests

test_page = requests.get("https://www.amazon.com/Bad-Bone-Down-Girl-Book-ebook/dp/B008P3IVQY?pf_rd_r=NT8GY9DE1MCK3DJF44GP&pf_rd_p=1a2142d2-7df1-467c-9640-2edd96993950&pd_rd_r=11a5adf8-2ad1-4fdb-b9cb-5b1e66850b19&pd_rd_w=VeIFN&pd_rd_wg=uKvic&ref_=pd_gw_rfyd")
resp = requests.get("https://www.amazon.com/Bad-Bone-Down-Girl-Book-ebook/dp/B008P3IVQY?pf_rd_r=NT8GY9DE1MCK3DJF44GP&pf_rd_p=1a2142d2-7df1-467c-9640-2edd96993950&pd_rd_r=11a5adf8-2ad1-4fdb-b9cb-5b1e66850b19&pd_rd_w=VeIFN&pd_rd_wg=uKvic&ref_=pd_gw_rfyd").content
soup = BeautifulSoup(resp, 'html.parser')
for i in soup.find_all("img"):
    print(i)