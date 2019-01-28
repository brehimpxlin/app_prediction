#!/usr/bin/env python
# coding: utf-8

# In[1]:


def preprocess(file):
    import numpy as np
    import datetime
    import os
    import shutil
    ''' Process dataset, extract features. '''

    # A dictionary for containing raw features.
    features = {}
    if 'saved_files' in os.listdir('.'):
        shutil.rmtree('saved_files')
    os.mkdir('saved_files')
    """ 
    读取数据集
    字段：日期时间，天气，wifi状态，位置坐标，电池电量，上次打开应用包名，本次打开应用包名，耳机状态，蓝牙状态，是否周末（未使用此字段）
    示例数据：2018/11/15 18:57:34,-1,-1,113.569399_22.372674,100,com.UCMobile,com.tencent.mm,0,0,0
    """
    # Read the data file.
    dataset = np.loadtxt(file,dtype='str',delimiter=',')

    """
    1. 提取直接的特征： 天气、WiFi状态、电池电量、上次打开应用包名、耳机状态、蓝牙状态、本次打开应用包名。 
    2. 对齐时间点： 丢弃“上次打开应用”、“本次打开应用”的第一行，丢弃其他属性的最后一行，
    使得“上次打开应用”变为“本次打开应用(curr_app)”，“本次打开应用”变为“下次打开应用(next_app)”。
    """
    # Get simple features.
    features['weather'] = [int(line[1]) for line in dataset[:-1]]
    features['wifi_status'] = [int(line[2]) for line in dataset[:-1]]
    features['battery'] = [int(line[4]) for line in dataset[:-1]]
    features['curr_app'] = [line[5].replace(':999','') for line in dataset[1:]]
    features['earphone'] = [int(line[7]) for line in dataset[:-1]]
    features['bluetooth'] = [int(line[8]) for line in dataset[:-1]]
    features['next_app'] = [line[6].replace(':999','') for line in dataset[1:]]

    """
    3. 填充缺失包名： 将“本次打开应用包名”中为null的记录全部填充为上一行中的“下次打开包名”。 
        如：
            2018/11/15 18:01:46,-1,-1,113.569399_22.372674,100,com.tencent.mobileqq,com.UCMobile,0,0,0
            2018/11/15 18:06:04,-1,-1,113.569399_22.372674,100,null,com.UCMobile,0,0,0
            第二行中的null用第一行的“com.UCMobile”填充。
    """
    for i in range(1,len(features['curr_app'])):
        if features['curr_app'][i] == 'null':
            features['curr_app'][i] = features['next_app'][i-1]

    """ 
    4. 时间 -> 周日期(weekday)、时段(hour)：
        1) 周日期即当天为周几。
        2) 时段精确到分钟。如18:30:25转化为18+30/60=18.5。
        如：由字段值“2018.12.25 18:30:25”可得到新特征
                weekday: 2
                hour: 18.5
    """
    # Get hour and weekday.
    date_time = [datetime.datetime.strptime(line[0],'%Y/%m/%d %H:%M:%S') for line in dataset[:-1]]
    features['hour'] = [float(dt.hour) for  dt in date_time]
    features['weekday'] = [dt.weekday() for dt in date_time]

    """
    5. 应用上次打开时间(last_open_time)：
        1) 根据特征： 本次打开应用包名、时间  
        2) 精确到秒，即计算本次打开的应用距离上次打开经过了多少秒。
        3) 数据集中第一次出现该应用的记录将这一特征的值填充为1296000（一个月）。
        如：
            2018/11/15 19:09:07,-1,-1,113.569299_22.372701,100,com.alibaba.android.rimet,com.tencent.mobileqq,0,0,0
            2018/11/15 19:09:24,-1,-1,113.569299_22.372701,100,com.tencent.mobileqq,com.alibaba.android.rimet,0,0,0
            2018/11/15 19:09:30,-1,-1,113.569299_22.372701,100,com.alibaba.android.rimet,com.UCMobile,0,0,0

            假设第一行为数据集中第一行，则此处应用“com.alibaba.android.rimet”的上次打开时间填充为1296000
            处理第三行时，应用“com.alibaba.android.rimet”的上次打开时间即 2018/11/15 19:09:30 - 2018/11/15 19:09:07 = 23
    """
    # Get last open time.
    features['last_open_time'] = []
    for counter, la in enumerate(features['curr_app']):
        last_open_time = 1296000
        for i in range(counter):
            if features['curr_app'][counter-1-i] == la:
                last_open_time = (date_time[counter] - date_time[counter-1-i]).seconds
                break
        features['last_open_time'].append(last_open_time)
    
    with open('./saved_files/last_open_time.txt', 'w') as f:
        written = set()
        length = len(features['curr_app'])
        for i in range(length):
            if features['curr_app'][length-1-i] not in written:
                f.write(str(features['curr_app'][length-1-i]) + ',')
                f.write(str(date_time[length-1-i]))
                f.write('\n')
                written.add(features['curr_app'][length-1-i])
    """
    6. 经度(longitude)、纬度(latitude)：根据坐标得到。
        如：
            根据字段值“113.569399_22.372674”得到新特征：
                longitude: 113.569399
                latitude: 22.372674
    """
    # Get  coordinates.
    coordinates =  [line[3] for line in dataset[:-1]]
    features['longitude'] = []
    features['latitude'] = []
    for c in coordinates:
        if c in 'unknown' or c in 'null':
            features['longitude'].append(np.nan)
            features['latitude'].append(np.nan)
        else:
            features['longitude'].append(float(c.split('_')[0]))
            features['latitude'].append(float(c.split('_')[1]))

    """
    7. 应用分类(categories)：
        1) 根据特征： 本次打开应用包名 
        2) 从top2000应用分类数据中查找分类
        3) 根据包名关键词分类
        如：
            根据字段值“com.tencent.mm”得到
                categories: “通讯社交”
    """
    # Get categories.
    import xlrd
    wb = xlrd.open_workbook('Top2000应用.xlsx') 
    sheet = wb.sheet_by_index(0) 
    app_cat = {}
    for i in range(sheet.ncols):
        app_cat[sheet.col(i)[0].value] = [cell.value for cell in sheet.col(i)[1:]]
    features['categories'] = []
    for a in features['curr_app']:
        if a in app_cat['包名']:
            features['categories'].append(app_cat['应用分类'][app_cat['包名'].index(a)])
        elif 'calculator' in a or 'dialer' in a or 'setting' in a or 'store' in a or 'weather' in a or 'blockchain' in a or 'clock' in a or 'assistant' in a             or 'center' in a or 'android' in a:
            features['categories'].append('系统工具')
        elif 'com.meizu.compaign' in a or 'com.meizu.flymelab' in a or 'com.meizu.filemanager' in a or 'com.meizu.notepaper' in a             or 'com.meizu.feedback' in a or 'com.meizu.flyme.toolbox' in a or 'com.meizu.mznfcpay' in a or  'com.meizu.safe' in a:
            features['categories'].append('系统工具')
        elif 'media' in a or 'video' in a or 'camera' in a or 'photo' in a or 'tv' in a or 'com.tencent.qqlive.player.meizu' in a:
            features['categories'].append('影音图像')
        elif 'health' in a or 'sport' in a:
            features['categories'].append('运动健康')
        elif 'email' in a or 'message' in a:
            features['categories'].append('通讯社交')
        else:
            features['categories'].append('其他')

    """
    8. 应用打开频率（frequency）：
        1) 根据特征： 本次打开应用包名
        2) 根据“本次应用包名”在整个数据集中出现的次数计算出。
    """
    # Get frequency.
    features['frequency'] = []
    apps_freq = {}
    for lp in features['curr_app']:
        if lp in apps_freq:
            apps_freq[lp] += 1
        elif lp == 'null':
            apps_freq['null'] = 0
        else:
            apps_freq[lp] = 1

    for lp in features['curr_app']:
        features['frequency'].append(apps_freq[lp]) 
    
    """
    导出frequency。
    """

    with open('./saved_files/frequency.txt', 'w') as f:
        for key, value in apps_freq.items():
            f.write(key+' '+str(value))
            f.write('\n')

    """
    9. 使用时长(use_time)：
        1) 根据特征： 本次打开应用包名、下次打开应用包名、时间 
        2) 计算当前打开应用在数据集中的平均使用时长，每一次使用时长由下一条记录的时间点减去当前记录的时间点得到。
        3) 该特征最终未使用。
    """    
# #     Get app average used time.
#     features['use_time'] = []
#     use_time = {}
#     for i in range(len(features['curr_app'])-1):
#         if features['curr_app'][i] in use_time:
#             use_time[features['curr_app'][i]].append((date_time[i+1] - date_time[i]).seconds)
#         else:
#             use_time[features['curr_app'][i]] = [(date_time[i+1] - date_time[i]).seconds]

#     for ut in use_time:
#         use_time[ut] = np.mean(use_time[ut])

#     for lp in features['curr_app']:
#         features['use_time'].append(use_time[lp])

    """
    10. 填充缺失值：
        1) 将耳机状态、蓝牙状态、WiFi状态的缺失值由-1改为0.5，方便归一化。
        2) 将应用打开频率的缺失值填充为中值。
        3) 将经度、纬度的缺失值填充为均值。
    """
    #  Handle missing values.
    earphone = [0.5 if e == -1 else e for e in features['earphone']]
    features['earphone'] = earphone

    bluetooth = [0.5 if b == -1 else b for b in features['bluetooth']]
    features['bluetooth'] = bluetooth

    wifi = [0.5 if ws == -1 or ws == 2 else ws for ws in features['wifi_status']]
    features['wifi_status'] = wifi

    longitude = []
    for l in features['longitude']:
        if l is not np.nan:
            longitude.append(l)
            break
    for c, v in enumerate(features['longitude']):
        if c == 0:
            continue
        if v is not np.nan:
            longitude.append(v)
        else:
            longitude.append(longitude[c-1])
    features['longitude'] = longitude
    
    latitude = []
    for l in features['latitude']:
        if l is not np.nan:
            latitude.append(l)
            break
    for c, v in enumerate(features['latitude']):
        if c == 0:
            continue
        if v is not np.nan:
            latitude.append(v)
        else:
            latitude.append(latitude[c-1])
    features['latitude'] = latitude
    
 
    """
    12. 处理连续值：
        1) 连续型特征有：电池电量、应用打开频率、经度、纬度、时段、耳机状态、蓝牙状态、wifi状态、周日期、上次打开时间。
        2) 对所有连续型特征进行归一化。
    """
    ''' Process continious values. '''
    # Normalization.
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_con = np.matrix([features['battery'], features['frequency'], features['longitude'], features['latitude'], features['hour'], features['earphone'], 
                       features['bluetooth'], features['wifi_status'], features['weekday'], features['last_open_time']]).T
    X_con = scaler.fit_transform(X_con)
    
    with open('./saved_files/min_max.txt','w') as f:
        for m in scaler.data_min_:
            f.write(str(m)+' ')
        f.write('\n')
        for r in scaler.data_range_:
            f.write(str(r)+' ')
    
    """
    13. 处理离散值：
        1) 离散型特征有：地理位置类型、应用分类、本次打开应用、天气。
        2) 对所有离散型特征进行label encoding（若要进行one hot encoding则剔除本次打开应用）。
    """
    ''' Process categorical values. '''
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
    lenc = LabelEncoder()
    enc = OneHotEncoder()
    lb = LabelBinarizer()
    # features['poi'] = LabelEncoder().fit_transform(features['poi'])
    cate = features['categories']
    ca = features['curr_app']
    wea = features['weather']
    features['categories'] = lenc.fit_transform(features['categories'])
    features['curr_app'] = lenc.fit_transform(features['curr_app'])
    features['weather'] = lenc.fit_transform(features['weather'])
    
    categories = {}
    for c,v in enumerate(cate):
        categories[v] = features['categories'][c]
    
    curr_app = {}
    for c,v in enumerate(ca):
        curr_app[v] = features['curr_app'][c]
    
    weather = {}
    for c,v in enumerate(wea):
        weather[v] = features['weather'][c]

    with open('./saved_files/categories.txt', 'w') as f:
        for key, value in categories.items():
            f.write(key+' '+str(value))
            f.write('\n')
        
    with open('./saved_files/curr_app.txt', 'w') as f:
        for key, value in curr_app.items():
            f.write(key+' '+str(value))
            f.write('\n')

    with open('./saved_files/weather.txt', 'w') as f:
        for key, value in weather.items():
            f.write(str(key)+' '+str(value))
            f.write('\n')

            
    
    # X_cat = np.matrix([features['categories'], features['poi'], features['weather']]).T
    # X_cat = enc.fit_transform(X_cat).toarray()
    X_cat = np.matrix([features['categories'], features['curr_app'], features['weather']]).T
#     print('X_cat: \n', X_cat)

    """
    14. 构建特征矩阵X：n×14。
    """
    ''' Prepare X. '''
#     print('\n*************** X MATRIX ***************\n')
#     print('X_con: ', X_con.shape)
#     print('X_cat: ', X_cat.shape)
    X = np.concatenate((X_con.T,X_cat.T)).T
#     print('X shape: ', X.shape)
#     print('X: \n', X)

    """
    15. 过滤低频应用以及一些无法处理的系统应用，筛选80%的高频应用作为待预测的类别。
    """
    ''' Select classes to predict. '''
    filtered = ['com.flyme.systemuiex', 'com.flyme.roamingpay', 'com.flyme.design', 'com.meizu.powersave', 'jp.co.hit_point.tabikaeru.meizu',                 'com.meizu.flyme.xtemui', 'com.meizu.mzsyncservice', 'com.meizu.account',  'com.meizu.perfui', 'com.meizu.flyme.easylauncher',                 'com.meizu.mzbasestationsafe', 'com.meizu.battery', 'com.meizu.share']
    for f in filtered:
        if f in apps_freq:
            apps_freq.pop(f)
    label = sorted(apps_freq, key=apps_freq.get, reverse=True)

    """
    16. 构建标签向量Y：n×1。
    """
    ''' Prepare Y. '''
#     print('\n*************** Y VECTOR ***************\n')

    Y = []
    options = set()
    others_num = 0
    for np in features['next_app']:
        if np not in label:
            Y.append('others') 
            others_num += 1
        else:
            Y.append(np)

#     print('#others: ',others_num)

    for y in Y:
        options.add(y)
#     print('Y: ', options)


    return (X, Y)


# In[2]:


import tensorflow as tf
import numpy as np

# tf.logging.set_verbosity(tf.logging.INFO)

# create input function
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# In[3]:


def train(file):
    """
    导出模型。
    """
    X, Y = preprocess(file)

    labels = list(set(Y))
    train_features = {
                        'battery': X.T.tolist()[0], 
                        'frequency': X.T.tolist()[1], 
                        'longitude': X.T.tolist()[2], 
                        'latitude': X.T.tolist()[3], 
                        'hour': X.T.tolist()[4],
                        'earphone': X.T.tolist()[5], 
                        'bluetooth': X.T.tolist()[6], 
                        'wifi_status': X.T.tolist()[7], 
                        'weekday': X.T.tolist()[8], 
                        'last_open_time': X.T.tolist()[9],
                        'categories': list(map(int, X.T.tolist()[10])),
                        'curr_app': list(map(int, X.T.tolist()[11])),
                        'weather': list(map(int, X.T.tolist()[12]))
                      }

    train_labels = Y

    # Define the feature column (describe how to use the features)
    battery = tf.feature_column.numeric_column('battery')
    frequency = tf.feature_column.numeric_column('frequency')
    longitude = tf.feature_column.numeric_column('longitude')
    latitude = tf.feature_column.numeric_column('latitude')
    hour = tf.feature_column.numeric_column('hour')
    earphone = tf.feature_column.numeric_column('earphone')
    bluetooth = tf.feature_column.numeric_column('bluetooth')
    wifi_status = tf.feature_column.numeric_column('wifi_status')
    weekday = tf.feature_column.numeric_column('weekday')
    last_open_time = tf.feature_column.numeric_column('last_open_time')
    categories = tf.feature_column.categorical_column_with_identity('categories',len(set(train_features['categories'])), 0)
    curr_app = tf.feature_column.categorical_column_with_identity('curr_app',len(set(train_features['curr_app'])),0)
    weather = tf.feature_column.categorical_column_with_identity('weather',len(set(train_features['weather'])),0)
    # Instantiate an estimator(2 hidden layer DNN with 64, 64 units respectively), passing the feature columns.
    estimator = tf.estimator.DNNClassifier(
        feature_columns=[battery, frequency, longitude, latitude, hour, earphone, bluetooth, wifi_status, weekday, last_open_time, 
                         tf.feature_column.embedding_column(categories, int(len(set(categories))), 'sqrtn'),
                         tf.feature_column.embedding_column(curr_app, int(len(set(curr_app))), 'sqrtn'),
                         tf.feature_column.embedding_column(weather, int(len(set(weather))), 'sqrtn')],

        # Two hidden layers of 64 nodes each.
        hidden_units=[64,64],
        # The model must choose between n classes.
        n_classes=len(labels),
        label_vocabulary=labels,
        optimizer=tf.train.ProximalAdagradOptimizer(
          learning_rate=0.01,
          l1_regularization_strength=0.001
        )
    )

    # Train the Model.
    estimator.train(
        input_fn=lambda: train_input_fn(train_features, train_labels, 512),
        steps=500)
    ''' Export trained model as a pb file. '''
    # Define inputs.
    feature_columns = [
        battery, frequency, longitude, latitude, hour, earphone, bluetooth, wifi_status, weekday, last_open_time, 
        tf.feature_column.embedding_column(categories, int(len(set(categories))), 'sqrtn'),
        tf.feature_column.embedding_column(curr_app, int(len(set(curr_app))), 'sqrtn'),
        tf.feature_column.embedding_column(weather, int(len(set(weather))), 'sqrtn')
    ]
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    # Build receiver function, and export.
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    export_dir = 'saved_model'
    estimator.export_savedmodel(export_dir_base=export_dir, serving_input_receiver_fn=serving_input_receiver_fn,
                                    strip_default_attrs=True)
    print('Model successfully trained.')
    pass


# In[4]:


def predict(data):
    import numpy as np
    import datetime
    import time
    start = time.time()

    features = {}
    data = data.split(',')
    features['weather'] = data[1]
    features['wifi_status'] = float(data[2])
    features['battery'] = float(data[4])
    features['curr_app'] = data[5].replace(':999','') 
    features['earphone'] = float(data[6])
    features['bluetooth'] = float(data[7])
    
    # Get hour and weekday.
    date_time = datetime.datetime.strptime(data[0],'%Y/%m/%d %H:%M:%S') 
    features['hour'] = float(date_time.hour) 
    features['weekday'] = date_time.weekday() 
    
    # Get last open time.
    lot = {}
    with open('./saved_files/last_open_time.txt') as f:
        for line in f:
            key, val = line.strip().split(',')
            lot[key] = datetime.datetime.strptime(val,'%Y-%m-%d %H:%M:%S')
    last_open_time = 1296000
    if features['curr_app'] in lot:
        last_open_time = (date_time - lot[features['curr_app']]).seconds
    features['last_open_time'] = last_open_time
    
    # Get  coordinates.
    coordinates =  data[3]
    if coordinates == 'unknown' or coordinates == 'null':
        features['longitude'] = np.nan
        features['latitude'] = np.nan
    else:
        features['longitude'] = float(coordinates.split('_')[0])
        features['latitude'] = float(coordinates.split('_')[1])

    # Get categories.
    import xlrd
    wb = xlrd.open_workbook('Top2000应用.xlsx') 
    sheet = wb.sheet_by_index(0) 
    app_cat = {}
    for i in range(sheet.ncols):
        app_cat[sheet.col(i)[0].value] = [cell.value for cell in sheet.col(i)[1:]]
    for a in features['curr_app']:
        if a in app_cat['包名']:
            features['categories'] = app_cat['应用分类'][app_cat['包名'].index(a)]
        elif 'calculator' in a or 'dialer' in a or 'setting' in a or 'store' in a or 'weather' in a or 'blockchain' in a or 'clock' in a or 'assistant' in a or 'center' in a             or 'android' in a:
            features['categories'] = '系统工具'
        elif 'com.meizu.compaign' in a or 'com.meizu.flymelab' in a or 'com.meizu.filemanager' in a or 'com.meizu.notepaper' in a             or 'com.meizu.feedback' in a or 'com.meizu.flyme.toolbox' in a or 'com.meizu.mznfcpay' in a or  'com.meizu.safe' in a:
            features['categories'] = '系统工具'
        elif 'media' in a or 'video' in a or 'camera' in a or 'photo' in a or 'tv' in a or 'com.tencent.qqlive.player.meizu' in a:
            features['categories'] = '影音图像'
        elif 'health' in a or 'sport' in a:
            features['categories'] = '运动健康'
        elif 'email' in a or 'message' in a:
            features['categories'] = '通讯社交'
        else:
            features['categories'] = '其他'
            
    # Get frequency.
    apps_freq = {}
    with open('./saved_files/frequency.txt') as f:
        for line in f:
            key, val = line.split()
            apps_freq[key] = int(val)

   
        if features['curr_app'] not in apps_freq:
             features['frequency'] = 0
        else:
             features['frequency'] = apps_freq[features['curr_app']]
    
    #  Handle missing values.
    earphone = 0.5 if features['earphone'] == -1 else features['earphone'] 
    features['earphone'] = earphone

    bluetooth = 0.5 if features['bluetooth'] == -1 else features['bluetooth']
    features['bluetooth'] = bluetooth

    wifi = 0.5 if features['wifi_status'] == -1 or features['wifi_status'] == 2 else features['wifi_status']
    features['wifi_status'] = wifi
    
    ''' Process continious values. '''
    X_con = [features['battery'], features['frequency'], features['longitude'], features['latitude'], features['hour'], features['earphone'], 
                       features['bluetooth'], features['wifi_status'], features['weekday'], features['last_open_time']]
    
    # Normalization.
    with open('./saved_files/min_max.txt') as f:
        first_line, second_line = list(f)
        data_mins = first_line.strip().split()
        data_ranges = second_line.strip().split()
    
    for c, v in enumerate(X_con):
        if v < float(data_mins[c]):
            X_con[c] = 0
        elif v > float(data_mins[c]) + float(data_ranges[c]):
            X_con[c] = 1
        elif float(data_ranges[c]) == 0:
            X_con[c] = 0
        else:
            X_con[c] = (v - float(data_mins[c])) / float(data_ranges[c])
    
    ''' Process categorical values. '''
    categories = {}
    with open('./saved_files/categories.txt') as f:
        for line in f:
            key, val = line.split()
            categories[key] = int(val)
    
    curr_app = {}
    with open('./saved_files/curr_app.txt') as f:
        for line in f:
            key, val = line.split()
            curr_app[key] = int(val)
            
    weather = {}
    with open('./saved_files/weather.txt') as f:
        for line in f:
            key, val = line.split()
            weather[key] = int(val)
    
    if features['categories'] in categories:
        features['categories'] = categories[features['categories']]
    else:
        features['categories'] = len(categories) + 2
        
    if features['curr_app'] in curr_app:
        features['curr_app'] = curr_app[features['curr_app']]
    else:
        features['curr_app'] = len(curr_app) + 2
        
    if features['weather'] in weather:
        features['weather'] = weather[features['weather']]
    else:
        features['weather'] = len(weather) + 2
        
    X_cat = [features['categories'], features['curr_app'], features['weather']]
        
    ''' Prepare X. '''
    X = np.matrix([X_con + X_cat])
    
    test_features = {
                    'battery': X.T.tolist()[0], 
                    'frequency': X.T.tolist()[1], 
                    'longitude': X.T.tolist()[2], 
                    'latitude': X.T.tolist()[3], 
                    'hour': X.T.tolist()[4],
                    'earphone': X.T.tolist()[5], 
                    'bluetooth': X.T.tolist()[6], 
                    'wifi_status': X.T.tolist()[7], 
                    'weekday': X.T.tolist()[8], 
                    'last_open_time': X.T.tolist()[9],
                     'categories': list(map(int, X.T.tolist()[10])),
                    'curr_app': list(map(int, X.T.tolist()[11])),
                    'weather': list(map(int, X.T.tolist()[12]))
                 }
    
    # 导入模型
    ''' Restore and predict. '''
    # Get path of saved models and choose the latest one.
    import os
    subdirs = [x for x in os.listdir('saved_model') if 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])

    # Restore model as a predictor. 
    predict_fn = tf.contrib.predictor.from_saved_model('saved_model/'+latest)

    examples = []
    for i in range(len(test_features['battery'])):
        feature = {}
        for key in test_features:
            if key in ['categories', 'curr_app', 'weather']:
                feature[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(test_features[key][i])]))
            else:
                feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[test_features[key][i]]))

        example = tf.train.Example(
            features=tf.train.Features(
                feature=feature
            )
        )
        examples.append(example.SerializeToString())
    
    # Make predictions.
    new_predictions = predict_fn({'inputs': examples})
    preds = []
    for counter, instance in enumerate(new_predictions['classes']):
        pre_cls = {}
        for index, cls in enumerate(instance):
            pre_cls[cls] = new_predictions['scores'][counter][index]
        preds.append(sorted(pre_cls, key=pre_cls.get, reverse=True)[:4])
    print(preds[0])
    
    end = time.time()
    print('Time used for prediction: ', end - start)
    
    import shutil
    for model_file in os.listdir('saved_model'):
        if model_file != latest:
            shutil.rmtree('./saved_model/' + model_file)
    pass


# In[5]:


"""
用以评估模型的函数。
评估方式：
    对与一个实例，根据预测参数num，返回num个预测结果，
    预测多个结果的方式：用predict_proba函数得到各个待预测类别的预测概率，根据概率排序，返回前num个
    在num个预测结果中若有一个与真实标签匹配，则此实例预测正确，记为1；否则此实例预测错误，记为0
    将数据集中所有实例的预测结果求平均值即得到此模型在该数据集的预测准确度。
"""

def evaluate(Y_pred_proba, classes, Y_test, num=4):
    ''' A function used to evaluate models. '''
    Y_pred = [[] for i in Y_pred_proba]
    for c,v in enumerate(Y_pred_proba):
        class_proba = {}
        for i in range(len(classes)):
            class_proba[classes[i]] = v[i]
        Y_pred[c] = sorted(class_proba, key=class_proba.get, reverse=True)[:num]
            
    result = []
    for i in range(len(Y_pred)):
        if Y_test[i] in Y_pred[i] and Y_test[i] is not 'others':
            result.append(1)
        else:
            result.append(0)
    
    accuracy = np.mean(result)
    return accuracy


# In[6]:


tf.logging.set_verbosity(tf.logging.ERROR)

import sys
operation = sys.argv[1]
data = sys.argv[2]
if operation is 't':
    train(data)
else:
    predict(data)

