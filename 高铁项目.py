# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 20:18:32 2018

@author: 安东
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from bokeh.plotting import figure,show,output_file
from bokeh.models import ColumnDataSource
import warnings
warnings.filterwarnings('ignore')
from pyecharts import Map
warnings.filterwarnings('ignore')
os.chdir('C:\\Users\\安东\\Desktop\\高铁数据3')
    #读取数据
data = pd.read_excel('高铁数据.xlsx')
data.rename(columns = {'经过站.1':'经过站发车时间','经过站.2':'经过站到达时间'},inplace = True)
'''
part 1 单个城市元素高铁数据表现力
'''
    #清洗重复数据(部分车次分上行下行)
data.index = data['车次']
data.drop(columns = '车次',inplace = True)
data.drop_duplicates(inplace = True)
data.reset_index(inplace = True)
    #创建车站转换城市函数
def f1(df_name,col_name):
    x = []
    for i in df_name[col_name].tolist():
        if len(i) == 5:
            x.append(i[:-1])
        if len(i) == 4:
            if i == '樟木头东':          
                x.append('东莞')
            elif i[-1] in ['东','南','西','北']:
                x.append(i[:-1])
            elif i == '上海虹桥':          
                x.append('上海')
            elif i == '洛阳龙门':
                x.append('洛阳') 
            elif i == '三  亚':
                x.append('三亚')
            elif i == '海  口':
                x.append('海口') 
            elif i[-1] not in ['东','南','西','北']:
                x.append(i)
        if len(i) == 3:
            if i == '常平东':          
                x.append('东莞')
            elif i[-1] in ['东','南','西','北']:
                x.append(i[:-1])      
            elif i[-1] not in ['东','南','西','北']:
                x.append(i)  
        if len(i) == 2:
            if i == '汉口':
                x.append('武汉')
            elif i == '武昌':
                x.append('武汉')
            elif i == '常平':
                x.append('东莞')
            else:
                x.append(i)
        if len(i) == 1:
            x.append(i)
    return x
data['经过站'] = f1(data,'经过站')
data.drop_duplicates(subset = ['车次','经过站'],inplace = True)#删除类似 同一车次 汉口和武汉的重复值 保留一个
    #计算各个车站日过车次数(包括普速车次)
counts_station = data.groupby('经过站')['车次'].count()
counts_station = pd.DataFrame({'车站':counts_station.index,'总车次':counts_station.values})
data['车次'] = data['车次'].apply(str)
    #筛选出高铁车次(包括：G,D,C)
crh_data = data[data['车次'].str.contains('C') | data['车次'].str.contains('D') | data['车次'].str.contains('G')]
    #得到各站高铁次数,和普速次数

crh_count_station = crh_data.groupby('经过站')['车次'].count()
crh_count_station = pd.DataFrame({'车站':crh_count_station.index,'高铁车次':crh_count_station.values})
counts_station = pd.merge(counts_station,crh_count_station,on = '车站',how = 'left').fillna(0)
counts_station['普快车次'] = counts_station['总车次'] - counts_station['高铁车次']
counts_station = counts_station.sort_values(by = '高铁车次',ascending = False).reset_index(drop =True)

counts_city = counts_station.copy()

    
    #合并部分城市不同火车站过车次数
counts_city = counts_city.groupby('车站')['总车次','高铁车次','普快车次'].sum()
'''
作为城际交通最方便的一种出行方式，高铁对于一个城市而言极为重要，哪些城市高铁车次非常多？
'''
counts_city.sort_values(by = '高铁车次',inplace = True,ascending= False)
counts_city.reset_index(inplace = True)

'''
part1--1  次数最多的前20城市
'''

    #绘制top20城市柱状图
from bokeh.models import HoverTool
counts_city_top20 = counts_city[:20]
counts_city_top20.columns = ['city','total','crh','normal']
stations = counts_city_top20['city'].tolist()[::-1]
source1 =  ColumnDataSource(counts_city_top20)
hover = HoverTool(tooltips = [('城市','@city'),
                              ('高铁车次','@crh')
                                ])
p1 = figure(title = '高铁车次最多TOP20城市',plot_width = 400,
            plot_height = 550,y_range = stations,
            tools=[hover,'box_select,reset,xwheel_zoom,pan,crosshair,save']
            )
p1.hbar(y = 'city',source = source1,color = '#02C39A',
        height = 0.2,left = 14.5,right = 'crh',alpha =0.6)
p1.xgrid.grid_line_color = None
p1.ygrid.grid_line_color = None
p1.xaxis.minor_tick_line_color = None
p1.yaxis.axis_line_color = None
p1.outline_line_color = None
p1.legend.location = "bottom_right"
p1.legend.orientation = "horizontal"

n = 19.2
from bokeh.models.annotations import Label
for k in counts_city_top20['crh'].tolist():
    if n> 0:
        label = Label(x = 1100,y = n-0.05,text = str(k).split('.')[0],
                       text_font_size = '9pt',text_color = '#898B8E')
        n = n-1               
        p1.add_layout(label)
output_file('高铁车次最多TOP20城市.html')     
show(p1)

print('中国共有大大小小的高铁车站%d个'%len(counts_city[counts_city['高铁车次'] != 0]))
'''
结论：1，高铁车次最多的top20城市中，珠江三角洲有（广州，深圳,东莞）
        '''
        
'''
part1--2 各城市高铁车次热力图
'''
lng_city = pd.read_excel('全国地市县区坐标\\全国地市县区坐标.xls')
lng_city_list = []
for city in lng_city['区县'].tolist():
    if len(city) > 2:
        lng_city_list.append(city[:-1])
    else:
        lng_city_list.append(city)
lng_city['区县'] = lng_city_list
counts_city_lng = pd.merge(lng_city,counts_city,left_on = '区县',right_on = '车站')
counts_city_lng = counts_city_lng[['车站','高铁车次','经度','纬度']]
crh_city = counts_city_lng[counts_city_lng['高铁车次'] != 0]
print('中国共有%d个城市和县开通高铁'%len(crh_city['车站'].unique().tolist()))
    #导出Excel查看热力图
counts_city_lng.to_excel('市县高铁车次情况.xlsx')
'''
part1--3 每个城市能直达多少城市（市县）
'''
checi = crh_data['车次'].unique().tolist()
crh_data1 = crh_data.copy()
crh_data1['经过站'] = f1(crh_data1,'经过站')
crh_data1['始发站'] = f1(crh_data1,'始发站')
crh_data1['终点站'] = f1(crh_data1,'终点站')
def f2():
    dic_checi = {}
    dic_station = {}
    for j in crh_city['车站'].tolist():
        dic_station[j] = []
    for i in checi:
        checi_station = crh_data1[crh_data1['车次'] == i]
        checi_count = checi_station['始发站'].tolist() + checi_station['终点站'].tolist() +checi_station['经过站'].tolist()
        dic_checi[i] = list(set(checi_count))        
        for j in crh_city['车站'].tolist():
            if j in dic_checi[i]:
                dic_station[j] = dic_station[j] + dic_checi[i]
    return dic_checi,dic_station
connect_city = f2()[1]
dic_checi = f2()[0]
    #查看每个城市和多少个城市能够直通
con_count = {}
for i in crh_city['车站'].tolist():
     con_count[i] = len(set(connect_city[i]))   
con_count_df = pd.DataFrame(con_count,index = range(len(con_count)))
con_count_df = con_count_df.T
con_count_df= con_count_df[0]
con_count_df = pd.DataFrame({'车站':con_count_df.index,'联通次数':con_count_df.values},
                            index = range(len(con_count_df)))
con_count_df.sort_values(by = '联通次数',inplace = True,ascending = False)
con_count_df.reset_index(inplace = True,drop = True)
con_count_df['联通次数'] = con_count_df['联通次数'] - 1 #去掉该城市本身
con_count_df.drop_duplicates('车站',inplace = True)
    #绘制联通次数柱状图
con_city_top20 = con_count_df[:20]
con_city_top20.columns = ['city','connect_count']
citys = con_city_top20['city'].tolist()[::-1]
source2 =  ColumnDataSource(con_city_top20)
hover = HoverTool(tooltips = [('城市','@city'),
                              ('直通城市(个)','@connect_count')
                                ])
p2 = figure(title = '直通城市个数最多TOP20城市',plot_width = 400,
            plot_height = 550,y_range = citys,
            tools=[hover,'box_select,reset,xwheel_zoom,pan,crosshair,save']
            )
p2.hbar(y = 'city',source = source2,color = '#02C39A',
        height = 0.2,left = 14.5,right = 'connect_count',alpha =0.6)
p2.xgrid.grid_line_color = None
p2.ygrid.grid_line_color = None
p2.xaxis.minor_tick_line_color = None
p2.yaxis.axis_line_color = None
p2.outline_line_color = None
p2.legend.location = "bottom_right"
p2.legend.orientation = "horizontal"

n = 19.2
for k in con_city_top20['connect_count'].tolist():
    if n> 0:
        label = Label(x = 370,y = n-0.05,text = str(k).split('.')[0],
                       text_font_size = '9pt',text_color = '#898B8E')
        n = n-1               
        p2.add_layout(label)
        
output_file('直通市县个数最多TOP20城市.html')     
show(p2)

#导出Excel查看热力图

con_count_df = pd.merge(con_count_df,lng_city,left_on = '车站',right_on = '区县')
con_count_df.to_excel('各城市直通城市.xlsx',index = False)


'''
part2-1  城市对高铁表现力
1,城市间通车次数
2，平均发车间隔
3，发车间隔分布稳定程度（标准差）
4，全天运行时长
5，平均到达时间
'''

def f3(df_name):#处理经过站到达时间格式
    df_name['经过站到达时间'] = df_name['经过站到达时间'].str.replace('当天00:02','第2日00:02')
    df_name['经过站到达时间'] = df_name['经过站到达时间'].str.replace('当天00:03','第2日00:03')
    #原始数据D13次和C7658次到达时间有误，做上述更改
    df_name['经过站到达时间'] = df_name['经过站到达时间'].str.replace('第2日','24:')
    df_name['经过站到达时间'] = df_name['经过站到达时间'].str.replace('第2日','24:')
    df_name['经过站到达时间'] = df_name['经过站到达时间'].str.replace('当天','00:')
    df_name['经过站到达时间'] = df_name['经过站到达时间'].str.replace('----','00:00')
    #df_name['经过站到达时间'] = df_name['经过站到达时间'].str.split('天').str[1]
    df_name['经过站到达时间'] = df_name['经过站到达时间'] +':00'
    return df_name

crh_data1 = f3(crh_data1)
crh_data1['经过站到达时间'][crh_data1['经过站到达时间']== '00:00:00'] = crh_data1['经过站发车时间']
def f4_1(df_name,col):#到达时间的‘第二日’处理
    df_name[col] = df_name[col].apply(str)
    df_name[col] = df_name[col].str.split(':',).str[0].apply(int)*60+df_name[col].str.split(':',).str[1].apply(int)*60+df_name[col].str.split(':',).str[2].apply(int)
f4_1(crh_data1,'经过站到达时间')    
def f4(df_name,col):#将时间字符串转换为每天该时间的分钟数 比如10:09:00 = 10*60+9 = 609
    df_name[col] = df_name[col].apply(str)
    df_name[col] = df_name[col].str.split(':',).str[0].apply(int)*60 + df_name[col].str.split(':',).str[1].apply(int)
    return df_name
f4(crh_data1,'经过站发车时间')

checi_df_list = []#得到每个车次中每个站的时间
n = 1
for checi_i in checi:
    checi_df = crh_data1[crh_data1['车次'] == checi_i][['经过站','经过站到达时间','经过站发车时间']]
    checi_df.drop_duplicates(subset = ['经过站'])#去除如：青岛北站（经f1转换后为青岛）与青岛站的重合
    checi_df.sort_values(by = '经过站到达时间',inplace = True)#排序时间
    checi_df_list.append(checi_df)
    print(n)
    n = n+1
    #x = checi_df

import itertools #导入排列组合模块
city_couple_list = [] #排列组合城市对以及对应的时间 如('南京', '马鞍山')对应时间(545, 562)
n = 1
for checi_df in checi_df_list:
    x = checi_df['经过站'].tolist()
    city_couple =  list(itertools.combinations(x, 2))
    y = checi_df['经过站到达时间'].tolist()
    city_couple_time = list(itertools.combinations(y, 2))
    z = checi_df['经过站发车时间'].tolist()
    city_couple_time1 = list(itertools.combinations(z, 2))
    city_couple_df = pd.DataFrame({'城市对':city_couple,'到达时间对':city_couple_time,
                                   '发车时间对':city_couple_time1})
    city_couple_list.append(city_couple_df)
    print(n)
    n+=1
city_couple = pd.concat(city_couple_list)
city_couple.to_excel('公交化城市对.xlsx',index = False)
city_couple = pd.read_excel('公交化城市对.xlsx')
    #提取城市 以及对应时间
city_couple.reset_index(drop = True,inplace = True)
city_couple['城市对'] = city_couple['城市对'].apply(str).str.replace('\'','')
city_couple['发车城市'] = city_couple['城市对'].str.split(',').str[0].str.split('(').str[1]
city_couple['到达城市'] = city_couple['城市对'].str.split(',').str[1].str.split(')').str[0]
city_couple['到达时间对'] = city_couple['到达时间对'].apply(str)
city_couple['发车时间'] = city_couple['发车时间对'].str.split(',').str[0].str.split('(').str[1]
city_couple['到达时间'] = city_couple['到达时间对'].str.split(',').str[1].str.split(')').str[0]
city_couple['到达时间'] = city_couple['到达时间'].apply(int)
city_couple['发车时间'] = city_couple['发车时间'].apply(int)

''' 指标计算
1,城市间通车次数
2，平均发车间隔
3，发车间隔分布稳定程度（标准差）
4，全天运行时长
5，平均到达时间
'''

    #groupby 计算五个指标
group1 = city_couple.groupby('城市对')['城市对'].count()#通车次数计算
group2 = city_couple.groupby('城市对')['发车时间'].max()#全天运行时长最大值
group3 = city_couple.groupby('城市对')['发车时间'].min()#全天运行时长最小值
#group4 = city_couple.groupby('城市对')['发车时间'].std()#发车时间稳定程度计算

group_by = pd.DataFrame({'通车次数':group1,'发车时间最大值':group2,'发车时间最小值':group3})
group_by.reset_index(inplace = True)
city_couple_re = pd.merge(city_couple,group_by,on = '城市对')
    #全天运行时长计算
city_couple_re['全天运行时长'] = city_couple_re['发车时间最大值'] - city_couple_re['发车时间最小值']
city_couple_re['到达时长'] = city_couple_re['到达时间'] - city_couple_re['发车时间']#到达时长计算
group5 = city_couple_re.groupby('城市对')[['到达时长']].mean()#平均到达时长计算
group6 = city_couple_re.groupby('城市对')[['到达时长']].min()
city_couple_re = pd.merge(city_couple_re,group5,left_on = '城市对',right_index = True)
city_couple_re = pd.merge(city_couple_re,group6,left_on = '城市对',right_index = True)
city_couple_re.rename(columns = {'到达到达':'到达时间','到达时长_x':'到达时长',
                                  '到达时长_y':'平均到达时长','到达时长':'最快到达时间'},
                                inplace =True)
city_couple_re['发车间隔'] = city_couple_re['全天运行时长'] / (city_couple_re['通车次数']-1)
 
        #取消‘发车时间稳定程度’指标，因为样本量（通车次数）不同，
        #所以std一定程度上不能比较两两城市对之间的稳定程度   
city_couple_re['城市对'] = city_couple_re['城市对'].str.replace(', ','-').str.replace('(','').str.replace(')','')
        #格式’城市对‘字段
city_bus_level = city_couple_re[['城市对','发车城市','到达城市','通车次数','到达时长',
                                 '全天运行时长','平均到达时长','发车间隔','最快到达时间']]
city_bus_level = city_bus_level.drop_duplicates(subset = ['城市对']).sort_values(by = '通车次数',ascending= False).reset_index(drop = True)

    #筛选出两两地级市之间的城市对
lng_city_list = []
for city in lng_city['地市'].tolist():
    if city[-1] == '市':
        lng_city_list.append(city[:-1])
    elif city[-2] == '地区':
        lng_city_list.append(city[:-2])
    elif city == '恩施土家族苗族自治州':
        lng_city_list.append('恩施')
    else:
        lng_city_list.append(city)
lng_city['地市'] = lng_city_list
lng_city1 = lng_city.copy()
for i in lng_city1.index:
    if lng_city1['地市'].loc[i] != lng_city1['区县'].loc[i]:
        lng_city1.drop(i,inplace = True)
lng_city1 = lng_city1[['地市','经度','纬度']]
city_bus_level_re = pd.merge(city_bus_level,lng_city1,left_on = '发车城市',right_on = '地市',how = 'left')
city_bus_level_re['到达城市'] = city_bus_level_re['到达城市'].str.replace(' ','')#去除空格
city_bus_level_re = pd.merge(city_bus_level_re,lng_city1,left_on = '到达城市',right_on = '地市',how = 'left')
city_bus_level_re.dropna(inplace = True)#只保留地级市城市对
city_bus_level_re.drop_duplicates(subset = ['城市对'],inplace = True)
for i in city_bus_level_re.index:
    if city_bus_level_re['发车城市'].loc[i] == city_bus_level_re['到达城市'].loc[i]:
        city_bus_level_re.drop(i,inplace = True)

    
''' 通车次数
    发车间隔(平均)          
    全天运行时长
    平均到达时长
    4个指标进行0-1标准化，计算总得分（标准化分值相加）
'''
def f5(col1,col2):#越大标注化分越高
    ma = city_bus_level_re[col2].max()
    mi = city_bus_level_re[col2].min()
    city_bus_level_re[col1] = (city_bus_level_re[col2] - mi) / (ma - mi)
    return city_bus_level_re

def f6(col1,col2):#越小标准化分越高
    ma = city_bus_level_re[col2].max()
    mi = city_bus_level_re[col2].min()
    city_bus_level_re[col1] = abs((city_bus_level_re[col2] - ma) / (ma - mi))
    return city_bus_level_re
city_bus_level_re = f5('通车次数_nor','通车次数')
city_bus_level_re = f5('全天运行时长_nor','全天运行时长')
city_bus_level_re = f6('发车间隔_nor','发车间隔')
city_bus_level_re = f6('平均到达时长_nor','平均到达时长')
city_bus_level_re['总得分'] = city_bus_level_re['通车次数_nor']+city_bus_level_re['全天运行时长_nor']+city_bus_level_re['发车间隔_nor']+city_bus_level_re['平均到达时长_nor']
city_bus_level_re.sort_values(by = '总得分',ascending = False,inplace = True)



    # 从双向城市对中筛选出单向城市对（比如：’广州-深圳‘和’深圳-广州‘，筛选出得分较高的一对）
x = city_bus_level_re['发车城市'].unique().tolist()
random_num = pd.DataFrame({'city':x,'y':list(np.random.rand(len(x)))})
city_bus_level_re = pd.merge(city_bus_level_re,random_num,left_on = '发车城市',
                             right_on = 'city',how = 'left')
city_bus_level_re = pd.merge(city_bus_level_re,random_num,left_on = '到达城市',
                             right_on = 'city',how = 'left')
city_bus_level_re['z'] = city_bus_level_re['y_x'] *city_bus_level_re['y_y']
city_bus_level_re.drop_duplicates('z','first',inplace = True)

'''
1、绘图
    柱状图：总得分、通车次数、最快达到时长、发车间隔、全天运行时长
    极轴图：总得分
    地图：1-8小时城市圈地图
2、计算
    每个省份中通高铁多少个县
    全国列车对 city_couple groupby
    普铁分布（普铁到高铁变化）
'''
    #绘制总得分top20雷达图
city_bus_level_re.reset_index(drop = True,inplace = True)
city_bus_level_re['全天运行时长'] = city_bus_level_re['全天运行时长']/60
total_top10 = city_bus_level_re[['城市对','总得分','通车次数_nor','全天运行时长_nor',
                                 '平均到达时长_nor','发车间隔_nor']][:20]
total_top10.T.to_excel('得分TOP20.xlsx',index = False)
total_top10_t = total_top10[:1]
fig = plt.figure(figsize=(7,28))
plt.subplots_adjust(wspace = 0.3,hspace = 0.7)
angels = np.linspace(0,2*np.pi,4,endpoint = False)
#angels = 角度

#####
n = 0
for i in total_top10['城市对'].tolist():
    #print(i)
    n += 1
    c = plt.cm.Greens_r(np.linspace(0,0.7,20))[n-1]
    axi = plt.subplot(10,2,n,projection = 'polar')
    yi = total_top10[['通车次数_nor','全天运行时长_nor',
                      '平均到达时长_nor','发车间隔_nor']][total_top10['城市对'] == i].T
    angels = np.linspace(0,2*np.pi,4,endpoint = False)
    scorei = total_top10['总得分'][total_top10['城市对'] == i]
    plt.polar(angels,yi,'-',linewidth = 1,color = c)
    axi.fill(angels,yi,alpha = 0.5,color = c)
    axi.set_thetagrids(np.arange(0,360,90),['通车次数','全天运行时长','平均到达时长','发车间隔'])
    axi.set_rgrids(np.arange(0.2,1.5,0.2),'--')
    plt.title('TOP%i %s: %.3f\n' %(n,i,scorei))

plt.savefig('公交化程度top20.png',dpi=400)


city_bus_level_re = city_bus_level_re[['城市对', '发车城市', '到达城市', '通车次数', '到达时长', 
                                       '全天运行时长', '平均到达时长', '发车间隔','最快到达时间',
                                       '经度_x', '纬度_x', '经度_y', '纬度_y','总得分']]
city_bus_level_re.to_excel('公交化程度总得分.xlsx',index = False)
print('全国所有地级市中互通高铁的城市对共有%d对' %(len(city_bus_level_re)))


city_bus_level_top = city_bus_level_re[city_bus_level_re['通车次数']>9]
def f7(df_name,col_name):
    df_name = city_bus_level_top[['城市对',col_name]]
    df_name.sort_values(by = col_name,inplace = True,ascending=False)
    df_name = df_name[:20]
    df_name.reset_index(drop = True,inplace = True)
    return df_name
total_top20 = f7('total_top20','总得分')
counts_top20 = f7('counts_top20','通车次数')
hour24_top20 = f7('hour_24_top20','全天运行时长')

def f8(df_name,col_name):
    df_name = city_bus_level_top[['城市对',col_name]]
    df_name.sort_values(by = col_name,inplace = True)
    df_name = df_name[:20]
    df_name.reset_index(drop = True,inplace = True)
    return df_name
space_top20 = f8('space_top20','发车间隔')
fast_arrive_top20 = f8('fast_arrive_top20','最快到达时间')

def f9(df,col,hovers,titles,ax,label):#
    df.columns = ['city',col]
    citys = df['city'].tolist()[::-1]
    source2 =  ColumnDataSource(df)
    hover = HoverTool(tooltips = [('城市','@city'),
                              (hovers,'@%s'%col)])
    p2 = figure(title = titles,plot_width = 400,
                plot_height = 550,y_range = citys,
                tools=[hover,'box_select,reset,xwheel_zoom,pan,crosshair,save']
                )
    p2.hbar(y = 'city',source = source2,color = '#02C39A',
            height = 0.2,left = 1,right = col,alpha =0.6)
    p2.xgrid.grid_line_color = None
    p2.ygrid.grid_line_color = None
    p2.xaxis.minor_tick_line_color = None
    p2.yaxis.axis_line_color = None
    p2.outline_line_color = None
    p2.legend.location = "bottom_right"
    p2.legend.orientation = "horizontal"
    p2.xaxis.axis_label = label
    n = 19.2
    for k in df[col].tolist():
        if n> 0:
            label = Label(x = ax,y = n-0.05,text = str(k)[:4],
                           text_font_size = '9pt',text_color = '#898B8E')
            n = n-1               
            p2.add_layout(label)       
    output_file('%s.html'%titles)       
    return show(p2)
f9(counts_top20,'level_count','通车次数','通车次数TOP20城市组对',270,'单位：次')
f9(space_top20,'space','发车间隔','发车间隔TOP20城市组对',8.5,'单位：分钟')
f9(fast_arrive_top20,'fast_arrive','最快到达时间','最快到达TOP20城市组对',19,'单位：分钟')
f9(hour24_top20,'hour_24','全天运行时长','全天运行时长TOP20城市组对',24,'单位：小时')
f9(total_top20,'total','总得分','总得分TOP20城市组队',4,'单位:分（总分：4分）')
counts_top20.to_excel('发车间隔.xlsx',index = False)

    #查看所有高铁车次始发站和终点站去向
start_end = crh_data1[['车次','始发站','终点站']].drop_duplicates(subset = ['车次'])

start_end['城市对'] = start_end['始发站'] + '-' +start_end['终点站']

start_group = start_end.groupby(['城市对'])[['城市对']].count()
start_end = pd.merge(start_end,start_group,left_on = '城市对',right_index = True,how = 'left')
start_end = pd.merge(start_end,lng_city1,left_on = '始发站',right_on = '地市',how = 'left')
start_end = pd.merge(start_end,lng_city1,left_on = '终点站',right_on = '地市',how = 'left')
start_end.dropna(inplace = True)

start_end['value'] =start_end['城市对_y'] / len(start_end)
for i in start_end.index:
    if start_end.loc[i]['始发站'] == start_end.loc[i]['终点站']:
        start_end.drop(i,inplace = True)
start_end.to_excel('始发-终点.xlsx',index= False)

shifa_pre = start_end[['始发站','终点站','城市对_x','城市对_y']].drop_duplicates()
x = shifa_pre.groupby(['始发站'])[['城市对_y']].sum()
y = shifa_pre.groupby(['终点站'])[['城市对_y']].sum()

'''part -2 城市圈划分
'''


'''
    part2-1单个城市城市圈表现力
'''
def f11(city,title):
    df = city_bus_level_re[['城市对','发车城市','到达城市','最快到达时间',
                            '总得分','经度_x','纬度_x','经度_y','纬度_y']]
    x = df[df['城市对'].str.contains(city)&(df['最快到达时间']<=120)]
    x['总得分']=x['总得分']*10
    x['color'] = '待分类'
    x['color'][x['最快到达时间']<=60] ='#00A896'
    x['color'][(x['最快到达时间']>60) & (x['最快到达时间']<=120)] = '#457B9D'
    colors = x[x['发车城市'] != city]['color'].tolist() + x[x['到达城市'] != city]['color'].tolist()
    citys = x[x['发车城市'] != city]['发车城市'].tolist() + x[x['到达城市'] != city]['到达城市'].tolist()
    lng = x[x['发车城市'] != city]['经度_x'].tolist() + x[x['到达城市'] != city]['经度_y'].tolist()
    lat = x[x['发车城市'] != city]['纬度_x'].tolist() + x[x['到达城市'] != city]['纬度_y'].tolist()
    #time = x[x['发车城市'] != city]['最快到达时间'].tolist() + x[x['到达城市'] != city]['最快到达时间'].tolist()
    size = x[x['发车城市'] != city]['总得分'].tolist() + x[x['到达城市'] != city]['总得分'].tolist()
    p = figure(title = title,plot_width = 800,
            plot_height = 800,
            tools=['box_select,reset,xwheel_zoom,ywheel_zoom,pan,crosshair,save']
            )
    p.circle(x = lng,y = lat,size = size,color = colors,
             fill_alpha = 0.8,line_width = 0.8,line_color = '#1C191D',
             legend = '绿色(一小时圈)蓝色(两小时圈)')
    p.circle_cross(x = x[x['发车城市'] == city]['经度_x'].tolist(),
             y = x[x['发车城市'] == city]['纬度_x'].tolist(),
             size = 26,
             fill_alpha = 0.8,line_width = 0.8,
             line_color = '#1C191D',color = '#F0374D',legend = '散点大小(%s与该市公交化得分)'%city)     
    for i,j,k,m in zip(lng,lat,citys,size):
        label = Label(x = i-0.09,y = j+0.165,text = k,
                           text_font_size = '9pt')
        label2 = Label(x = i-0.09,y = j-0.044,text = str(m/10)[:4],
                           text_font_size = '9pt')
        p.add_layout(label)
        p.add_layout(label2)
    label3 = Label(x =  x[x['发车城市'] == city]['经度_x'].tolist()[0]-0.09,
                   y =  x[x['发车城市'] == city]['纬度_x'].tolist()[0]+0.165,
                   text = city,text_font_size = '9pt')
    p.add_layout(label3)
    p.legend.location = "bottom_right"
    p.legend.background_fill_alpha = 0.05
    p.legend.border_line_color = None
    p.background_fill_alpha  = 0.05
    df_y = pd.DataFrame({'LNG':lng,'lat':lat,'city':citys})
    return show(p),df_y,x

f11('成都','成都城市圈')
f11('上海','上海城市圈')
f11('北京','北京城市圈')
f11('济南','济南城市圈')
f11('郑州','郑州城市圈')
f11('西安','西安城市圈')
f11('广州','广州城市圈')
f11('福州','福州城市圈')
f11('武汉','武汉城市圈')
f11('杭州','杭州城市圈')
f11('南京','南京城市圈')


'''part2-2单个城市直达城市个数
'''
    #利用lng_city1筛选出city_bus_level位地级市的城市
single_city = city_bus_level[['城市对','发车城市','到达城市','最快到达时间']]
single_city['到达城市'] = single_city['到达城市'].str.strip()
single_city = pd.merge(single_city,lng_city1,left_on = '发车城市',right_on = '地市',how = 'left')
single_city = pd.merge(single_city,lng_city1,left_on = '到达城市',right_on = '地市',how = 'left')
single_city.dropna(inplace = True)

    #赋值随机数给城市 删掉例如南京-杭州 或 杭州-南京中的一个
x = lng_city1['地市'].tolist()
random_num = pd.DataFrame({'city':x,'y':list(np.random.rand(len(x)))})
random_num.drop_duplicates('city',inplace = True)
single_city = pd.merge(single_city,random_num,left_on = '发车城市',
                             right_on = 'city',how = 'left')
single_city = pd.merge(single_city,random_num,left_on = '到达城市',
                             right_on = 'city',how = 'left')
single_city['z'] = single_city['y_x'] *single_city['y_y']
single_city.drop_duplicates('z','first',inplace = True)
    #得到每个城市直通的个数
counts = []
citys = []
for city in lng_city1['地市'].tolist():
    count = len(single_city[single_city['城市对'].str.contains(city)])
    citys.append(city)
    counts.append(count)
single_city_re = pd.DataFrame({'城市':citys,'直达个数':counts})
single_city_re= single_city_re[single_city_re['直达个数'] != 0]
    #合并counts_city
single_city_re = pd.merge(single_city_re,counts_city,left_on = '城市',right_on = '车站')
single_city_re= single_city_re[['城市','高铁车次','直达个数']]
single_city_re.drop_duplicates('城市',inplace =True)
'''part2-2单个城市1h、2h城市圈个数
'''
citys = single_city_re['城市'].tolist()
h1_h2 = []#短途
h2_h4 = []#中途
h4_h8 = []#长途
h8_ = []#特长途

for city in citys:
    df = single_city[['城市对','发车城市','到达城市','最快到达时间']]
    x = df[df['城市对'].str.contains(city)]
    len1 = len(x[(x['最快到达时间'] > 0) & (x['最快到达时间'] <= 120)])
    h1_h2.append(len1)
    len2 = len(x[(x['最快到达时间'] > 120) & (x['最快到达时间'] <= 240)])
    h2_h4.append(len2)
    len3 = len(x[(x['最快到达时间'] > 240) & (x['最快到达时间'] <= 480)])
    h4_h8.append(len3)
    len4 = len(x[(x['最快到达时间'] > 480)])
    h8_.append(len4)

single_city_re['1h_2h'] = h1_h2
single_city_re['2h_4h'] = h2_h4
single_city_re['4h_8h'] = h4_h8
single_city_re['8h_'] = h8_
'''结论：
    1、在1-2h内
    
'''
'''
part2-3 各个城市高铁站数量
'''
city_station = pd.DataFrame({'车站':crh_data['经过站'].unique().tolist()})
city_station['城市'] = f1(city_station,'车站')
city_station = pd.merge(city_station,lng_city,left_on = '城市',right_on = '区县')
city_station.drop_duplicates('车站',inplace = True)
city_station_count = city_station.groupby('地市')[['地市']].count()
city_station_count.reset_index(inplace = True,drop = False)
single_city_re = pd.merge(single_city_re,city_station_count,left_on = '城市',right_index = True)

'''
part2_4 每个城市最大的辐射半径（距离）
'''


EARTH_REDIUS = 6378.137#地球半径
def rad(d):#角度制转换弧度制
    return d * np.pi / 180.0

def getDistance(lat1, lng1, lat2, lng2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * np.arcsin(np.sqrt(pow(np.sin(a/2), 2) + np.cos(radLat1) * np.cos(radLat2) * pow(np.sin(b/2), 2)))
    s = s * EARTH_REDIUS
    return s
single_city['直线距离'] = getDistance(single_city['经度_x'],single_city['纬度_x'],single_city['经度_y'],single_city['纬度_y'])
distance_max =[]
distance_sum =[]
fast_time_sum = []
for city in single_city_re['城市'].tolist():
    dis_max = single_city[single_city['城市对'].str.contains(city)]['直线距离'].max()
    distance_max.append(dis_max)
    dis_sum = single_city[single_city['城市对'].str.contains(city)]['直线距离'].sum()
    distance_sum.append(dis_sum)
    time_sum = single_city[single_city['城市对'].str.contains(city)]['最快到达时间'].sum()
    fast_time_sum.append(time_sum)
single_city_re['最远距离'] = distance_max
single_city_re['距离之和'] = distance_sum
single_city_re['时间之和'] = fast_time_sum
single_city_re['平均辐射半径'] = single_city_re['距离之和']/single_city_re['直达个数']

'''part2-5 综合得分计算
1、 高铁车次*0.5
2、直达个数*0.4
3、距离之和*0.1
'''
    #三指标加权计算
def f12(df,col1,col2):#越大标注化分越高
    ma = df[col2].max()
    mi = df[col2].min()
    df[col1] = (df[col2] - mi) / (ma - mi)
    return df
single_city_re = f12(single_city_re,'高铁车次_nor','高铁车次')
single_city_re = f12(single_city_re,'直达个数_nor','直达个数')
single_city_re = f12(single_city_re,'距离之和_nor','距离之和')

single_city_re['综合得分'] = single_city_re['高铁车次_nor']*0.5 +single_city_re['直达个数_nor']*0.4 +single_city_re['距离之和_nor']*0.1
single_city_re.sort_values(by = '综合得分',ascending= False,inplace = True)
single_city_re.reset_index(drop = True,inplace = True)
single_city_re.drop_duplicates('城市',inplace = True)  
    #绘制TOP20雷达图
total_top20 = single_city_re[['城市','综合得分','高铁车次_nor','直达个数_nor',
                                 '距离之和_nor']][:20]

total_top20_t = total_top20[:1]
fig = plt.figure(figsize=(7,28))
plt.subplots_adjust(wspace = 0.3,hspace = 0.7)
angels = np.linspace(0,2*np.pi,3,endpoint = False)

n = 0
for i in total_top20['城市'].tolist():
    #print(i)
    n += 1
    c = plt.cm.Greens_r(np.linspace(0,0.7,20))[n-1]
    axi = plt.subplot(10,2,n,projection = 'polar')
    yi = total_top20[['高铁车次_nor','直达个数_nor',
                      '距离之和_nor']][total_top20['城市'] == i].T
    angels = np.linspace(0,2*np.pi,3,endpoint = False)
    scorei = total_top20['综合得分'][total_top20['城市'] == i]
    plt.polar(angels,yi,'-',linewidth = 1,color = c)
    axi.fill(angels,yi,alpha = 0.5,color = c)
    axi.set_thetagrids(np.arange(0,360,120),['高铁车次','直达个数','距离之和'])
    axi.set_rgrids(np.arange(0.2,1.5,0.2),'--')
    plt.title('TOP%i %s: %.3f\n' %(n,i,scorei))
plt.savefig('高铁城市表现力top20.png',dpi=400)

    #城市表现力划分以及城市群
city_group = pd.read_excel('城市圈划分.xlsx')
city_group['city'] = city_group['city'].str.split('、')
city_group_list = []
for i in city_group.index:
    x = pd.DataFrame({'城市':city_group.loc[i]['city'],'城市群':city_group.loc[i]['city_circle']})
    city_group_list.append(x)
city_group = pd.concat(city_group_list)
single_city_area =single_city_re[['城市', '高铁车次', '直达个数','综合得分','平均辐射半径']]
single_city_area.sort_values(by = '高铁车次',inplace =True,ascending= False)
single_city_area = pd.merge(single_city_area,lng_city1,left_on = '城市',
                            right_on = '地市',how = 'left')
single_city_area = pd.merge(single_city_area,city_group,on = '城市',how = 'left')
single_city_area.drop_duplicates('城市','first',inplace =True)
single_city_area.fillna('未划分',inplace =True)
single_city_area['color'] = '未分类'
colors = ['#F25F5C','#FFE066','#EF233C','#247BA0','#70C1B3','#264653'
         ,'#2A9D8F','#E9C46A','#F4A261','#E76F51','#50514F'
         ,'#F0F3BD','#8D99AE','#D8E2DC','#F4ACB7']

for i,j in zip(single_city_area['城市群'].unique().tolist(),colors):
    single_city_area['color'][single_city_area['城市群'] == i] =j

single_city_area = single_city_area[['城市', '高铁车次', '直达个数', '综合得分', 
                                     '平均辐射半径', '经度', '纬度','城市群','color']]
single_city_area['size'] = single_city_area['综合得分']*50

single_city_area.columns = ['city','crh_count','city_count','score',
                            'r_mean','lng','lat','city_circle','color','size']
source2 = ColumnDataSource(data = single_city_area)
hover = HoverTool(tooltips = [('城市','@city'),
                              ('高铁次数','@crh_count'),
                              ('直达个数','@city_count'),
                              ('平均辐射半径','@r_mean')])
p3 = figure(plot_width = 1200,plot_height = 800,title = '城市群划分',
            tools=[hover,'box_select,reset,xwheel_zoom,pan,save,crosshair'])
p3.circle(x = 'lng',y = 'lat',size = 'size',source = source2,
          fill_color = 'color',fill_alpha = 0.8,line_width =0.8,
          line_color = '#1C191D')
legend = single_city_area[['city_circle','color']].drop_duplicates()
legend['x'] = 90
legend['y'] = range(22,37)[::-1]
source3 = ColumnDataSource(data = legend)
p3.circle(x = 'x',y ='y' ,size = 15,source = source3,
              fill_color = 'color',fill_alpha = 0.8,line_width =0.8,
              line_color = '#1C191D')
for y,k in zip(legend['y'].tolist(),legend['city_circle'].tolist()):
    labels= Label(x = 91,y = y-0.3,text = k,text_font_size = '9pt')  
    p3.add_layout(labels) 
p3.xaxis.axis_label = '经度'
p3.yaxis.axis_label = '纬度'
p3.outline_line_color = None
show(p3)


'''
part3 各高铁站表现力
'''
'''
part3-1 高铁站规模
'''
    #重新读取数据
data = pd.read_excel('高铁数据.xlsx')
data.rename(columns = {'经过站.1':'经过站发车时间','经过站.2':'经过站到达时间'},inplace = True)
data.index = data['车次']
data.drop(columns = '车次',inplace = True)
data.drop_duplicates(inplace = True)
data.reset_index(inplace = True)
crh_data = data[data['车次'].str.contains('C') | data['车次'].str.contains('D') | data['车次'].str.contains('G')]
    
station = pd.read_excel('高铁站站台数据.xlsx')
station['车站'] = station['车站'].str.replace('站','')
station['台数'] = station['规模'].str.split('台').str[0]
station['线数'] = station['规模'].str.split('台').str[1].str[:2]
station.columns = ['stations','plat_line','crh_line','platform','line']
station['line'] = station['line'].apply(float)
station['platform'] = station['platform'].apply(float)
station['score'] = station['line'] * 0.3 +station['platform']*0.3 +station['crh_line']*0.4
#股道权重大一点的原因是站台满足不了股道可以新建，需要新增股道比新建站台困难
station.sort_values(by = 'score',ascending = False,inplace = True)

station1 = station[:20]
station1['plat_line'] = station1['plat_line'].str.replace('线','股')
station1['crh_line'] =  station1['crh_line'].apply(str)
station1['label'] = station1['plat_line']+station1['crh_line']+'线'

    #绘制车站规模散点图
stations = station1['stations'].tolist()[::-1]
source4 = ColumnDataSource(station1)
hover = HoverTool(tooltips = [('车站','@stations'),
                              ('站台数','@platform'),
                              ('到发线','@line'),
                              ('高铁线路','@crh_line')])
p4 = figure(title = '全国大型高铁站规模',plot_width = 400,
            plot_height = 550,y_range = stations,x_range = [0,25],
            tools=[hover,'box_select,reset,xwheel_zoom,pan,save,crosshair']
            )
#p4.circle(y = 'stations',x = 'score',source = source4,size = 'size',
 #         color = '#05668D',line_color = 'black',legend = '车站规模',
  #        line_width = 1,fill_alpha = 0.6)
p4.hbar(y = 'stations',source = source4,color = '#02C39A',
        height = 0.2,left = 2.5,right = 'score',alpha =0.6)
#p1.xaxis.axis_label = '车站股道数'
n = 19.2
for i in station1['label'].tolist():
    label = Label(x = 18,y = n,text = i,text_font_size ='9pt')
    p4.add_layout(label)
    n -= 1
p4.xgrid.grid_line_color = None
p4.ygrid.grid_line_color = None
p4.xaxis.minor_tick_line_color = None
p4.yaxis.axis_line_color = None
p4.outline_line_color = None
p4.legend.location = "bottom_right"
output_file('高铁站规模Top20.html')
show(p4)

'''
part3-2 高铁站日过车次数
'''
station_counts = data.groupby('经过站')['车次'].count()
station_counts = pd.DataFrame({'站名':station_counts.index,'总车次':station_counts.values})
data['车次'] = data['车次'].apply(str)
#有的一趟车分为两个车次（上下行的问题）,删除这些数据
crh_data = data[data['车次'].str.contains('C') | data['车次'].str.contains('D') | data['车次'].str.contains('G')]

crh_count = crh_data.groupby('经过站')['车次'].count()
crh_count = pd.DataFrame({'站名':crh_count.index,'高铁车次':crh_count.values})

station_counts = pd.merge(station_counts,crh_count,on = '站名',how = 'left')
station_counts['普快车次'] = station_counts['总车次'] - station_counts['高铁车次']
station_counts = station_counts.sort_values(by = '高铁车次',ascending = False).reset_index(drop =True)
station_counts.fillna(0,inplace = True)
    #绘制高铁站日发车次Top20
from bokeh.core.properties import value
counts_top20 = station_counts[:20]
counts_top20.columns = ['stations','total','crh','normal']
stations = counts_top20['stations'].tolist()[::-1]
source2 =  ColumnDataSource(counts_top20)
hover = HoverTool(tooltips = [('车站','@stations'),
                              ('日车次','@total')
                                ])
p5 = figure(title = '高铁站日发车次Top20',plot_width = 400,
            plot_height = 550,y_range = stations,
            tools=[hover,'box_select,reset,xwheel_zoom,pan,crosshair']
            )
crh_nor = ['crh','normal']
colors = ['#02C39A','#05668D']
hbar_stack1 = p5.hbar_stack(crh_nor,y = 'stations',source = source2,
                          color = colors,height = 0.2,alpha =0.6,
                          legend = [value(x)  for x in crh_nor],name =crh_nor)
p5.xgrid.grid_line_color = None
p5.ygrid.grid_line_color = None
p5.xaxis.minor_tick_line_color = None
p5.yaxis.axis_line_color = None
p5.outline_line_color = None
p5.legend.location = "bottom_right"
p5.legend.orientation = "horizontal"
output_file('高铁站日发车次Top20.html')
show(p5)
'''
part3-3 高铁站最高峰发车数（单指高铁车次）
'''
station_time = crh_data[['车次','经过站','经过站到达时间','经过站发车时间']]
station_time['经过站到达时间'] = station_time['经过站到达时间'].str.replace('日','天')
station_time['经过站到达时间'] = station_time['经过站到达时间'].str.split('天').str[1]
station_time['经过站到达时间'] = ['%s:00' % i for i in station_time['经过站到达时间']]
    #修改后的数据 部分时间有异常 ('---:00'及'00:00:00')
station_time['经过站到达时间'][station_time['经过站到达时间']== '00:00:00'] = station_time['经过站发车时间']
station_time['经过站到达时间'][station_time['经过站到达时间']== '----:00'] = station_time['经过站发车时间']
    #得到发车时间段        
station_time['发车小时段'] = station_time['经过站发车时间'].str.split(':',2).str[0]
station_time['发车分钟段'] = station_time['经过站发车时间'].str.split(':',2).str[1]
    #最高发车小时段
group1 = station_time.groupby(['经过站','发车小时段'])[['车次']].count().reset_index()
station_time = pd.merge(station_time,group1,on = ['经过站','发车小时段'])
station_time.rename(columns = {'车次_y':'该小时段发车总数','车次_x':'车次'},inplace =True)
station_time['经过站发车时间'] = station_time['经过站发车时间'].str.split(':').str[0].apply(int)*60 + station_time['经过站发车时间'].str.split(':').str[1].apply(int)
    #全天运行时长
group2 = station_time[['经过站','发车小时段']].drop_duplicates()
group2 = group2.groupby(['经过站']).count().reset_index()
group2.rename(columns = {'发车小时段':'全天运行小时'},inplace = True)
station_counts = pd.merge(station_counts,group2,left_on = '站名',right_on = '经过站')
'''part3-4平均发车间隔
'''
station_time.sort_values(by = ['经过站','该小时段发车总数'],ascending = False,inplace = True)
group3 = station_time.groupby(['经过站'])['该小时段发车总数'].max().reset_index()
station_counts = pd.merge(station_counts,group3,left_on = '站名',
                          right_on = '经过站',how = 'left')
station_counts['平均发车间隔'] = station_counts['全天运行小时']*60/station_counts['高铁车次']

station_counts = pd.merge(station_counts,station,left_on = '站名',right_on = 'stations',how = 'left')
station_counts = station_counts[['站名','高铁车次', '全天运行小时',
                   '该小时段发车总数','平均发车间隔','score']]
station_counts.rename(columns = {'该小时段发车总数':'最高峰发车数'},inplace =True)
station_counts_re = station_counts.copy()
station_counts_re.dropna(inplace = True)
'''
part3-5 z-score 计算综合得分
'''

def f13(col1,col2):#越大标注化分越高
    ma = station_counts_re[col2].max()
    mi = station_counts_re[col2].min()
    station_counts_re[col1] = (station_counts_re[col2] - mi) / (ma - mi)
    return station_counts_re
def f14(col1,col2):#越小标准化分越高
    ma = station_counts_re[col2].max()
    mi = station_counts_re[col2].min()
    station_counts_re[col1] = abs((station_counts_re[col2] - ma) / (ma - mi))
    return station_counts_re
f13('高铁车次_z','高铁车次')
f13('运行小时_z','全天运行小时')
f13('高峰发车数_z','最高峰发车数')
f13('score_z','score')
f14('平均发车间隔_z','平均发车间隔')
def f15(df_name):
    x = df_name
    x['total_score'] = x['高铁车次_z']*0.4+x['平均发车间隔_z']*0.2+x['score_z']*0.2+x['运行小时_z']*0.1+x['高峰发车数_z']*0.1
    df_name = x
    return df_name
station_counts_re = f15(station_counts_re)    
station_counts_re.sort_values(by = 'total_score',inplace = True,ascending = False)











