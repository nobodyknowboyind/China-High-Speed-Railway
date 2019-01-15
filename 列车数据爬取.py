# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 08:50:41 2018

@author: 安东
"""

from lxml import etree
import requests
import pandas as pd
import os
os.chdir('C:\\Users\\安东\\Desktop\\高铁数据3')
    #读取数据网址
citys_file = open('市县级地址.txt','r')
def openurl(url):
    head={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.92 Safari/537.36',}
    res=requests.get(url,headers=head)
    res.encoding='gb2312'
    text=res.text
    return text
def parseurl(text):
    html=etree.HTML(text)
    tr=html.xpath('//tr[@onmouseover="this.bgColor=\'#E6F2E7\';"]')
    checi = []
    leixing =[]
    shifazhan = []
    shifashijian = []
    jingguozhan = []
    jingguozhandaodashijian = []
    jingguozhanfacheshijian = []
    zhongdianzhan = []
    daodashijian = []
    for each in tr:
        checi.append(each.xpath('./td[1]/a/b/text()')[0])
        leixing.append(each.xpath('./td[2]/text()')[0])
        shifazhan.append(each.xpath('./td[3]/text()')[0])
        shifashijian.append(each.xpath('./td[4]/text()')[0])
        jingguozhan.append(each.xpath('./td[5]/text()')[0])
        jingguozhandaodashijian.append(each.xpath('./td[6]/text()')[0])
        jingguozhanfacheshijian.append(each.xpath('./td[7]/text()')[0])
        zhongdianzhan.append(each.xpath('./td[8]/text()')[0])
        daodashijian.append(each.xpath('./td[9]/text()')[0])
    data = pd.DataFrame({'车次':checi,'类型':leixing,'始发站':shifazhan,
                         '始发时间':shifashijian,'经过站':jingguozhan,
                         '经过站到达时间':jingguozhandaodashijian,
                         '经过站发车时间':jingguozhanfacheshijian,
                         '终点站':zhongdianzhan,'到达时间':daodashijian})
    #print(data)
    #print(len(data))
    return data
lines = citys_file.readlines()
    
def main():
    n = 0
    lines2 = []
    data_list = []
    for line in lines:
        line = line.strip('\n')
        lines2.append(line) 
    for j in lines2:
        n = n+1
        print(n)
        url = j
        text=openurl(url)
        parseurl(text)
        data_list.append(parseurl(text))
    return data_list
if __name__=='__main__':
    data_list = main()

data = pd.concat(data_list)
data.to_excel('高铁数据.xlsx',index = False)