import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

data_root = './data'

# Data Cleaning
def convertUnknowToNan(df, attr_val):
    for _, row in attr_val[attr_val['Meaning'].isin(['unknown', 'unknown / no main age detectable'])].iterrows():
        key, val = row['Attribute'], dict()
        for x in row['Value'].split(', '):
            val[int(x)] = np.nan

        if key in df.columns:
            df = df.replace({key: val})
    return df


def removeNan(df, col_null_names=['ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3', 'ALTER_KIND4', 'KBA05_BAUMAX', 'KK_KUNDENTYP', 'TITEL_KZ', 'EINGEFUEGT_AM']):
    return df.drop(col_null_names, axis=1)


def cleanCategorical(df, modes, create_modes=False, replaceNanVal=-1):
    # replace 'X' to NaN
    replace_dict = {'CAMEO_DEU_2015': {'XX': np.nan}, 'CAMEO_DEUG_2015': {'X': np.nan}, 'CAMEO_INTL_2015': {'XX': np.nan}}
    for key, val in replace_dict.items():
        if key in df.columns:
            df = df.replace({key: val})

    # CAMEO_DEU_2015
    cam = pd.get_dummies(df['CAMEO_DEU_2015'], prefix='CAMEO_DEU_2015', dummy_na=False).astype('int64')
    df = pd.concat([df, cam], axis=1)

    # CAMEO_DEUG_2015
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].map(lambda x: int(x), na_action='ignore')
    if create_modes:
        modes['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].mode().values[0]
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].fillna(modes['CAMEO_DEUG_2015'])
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].astype('int64')

    # CAMEO_INTL_2015
    df['CAMEO_INTL_2015_1'] = df['CAMEO_INTL_2015'].map(
        lambda x: int(int(x)/10), na_action='ignore')
    if create_modes:
        modes['CAMEO_INTL_2015_1'] = df['CAMEO_INTL_2015_1'].mode().values[0]
    df['CAMEO_INTL_2015_1'] = df['CAMEO_INTL_2015_1'].fillna(modes['CAMEO_INTL_2015_1'])
    df['CAMEO_INTL_2015_1'] = df['CAMEO_INTL_2015_1'].astype('int64')

    df['CAMEO_INTL_2015_2'] = df['CAMEO_INTL_2015'].map(lambda x: int(x) % 10, na_action='ignore')
    if create_modes:
        modes['CAMEO_INTL_2015_2'] = df['CAMEO_INTL_2015_2'].mode().values[0]
    df['CAMEO_INTL_2015_2'] = df['CAMEO_INTL_2015_2'].fillna(modes['CAMEO_INTL_2015_2'])
    df['CAMEO_INTL_2015_2'] = df['CAMEO_INTL_2015_2'].astype('int64')

    # D19_LETZTER_KAUF_BRANCHE
    d19 = pd.get_dummies(df['D19_LETZTER_KAUF_BRANCHE'], prefix='D19_LETZTER_KAUF_BRANCHE', dummy_na=False)
    d19 = d19.astype('int64')
    df = pd.concat([df, d19], axis=1)

    # OST_WEST_KZ
    if create_modes:
        modes['OST_WEST_KZ'] = df['OST_WEST_KZ'].mode().values[0]
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].fillna(modes['OST_WEST_KZ'])
    df = pd.concat([df, pd.get_dummies(df['OST_WEST_KZ'], prefix='OST_WEST_KZ', dummy_na=False)], axis=1)
    df['OST_WEST_KZ_W'] = df['OST_WEST_KZ_W'].astype('int64')

    df.drop(['CAMEO_DEU_2015', 'CAMEO_INTL_2015', 'OST_WEST_KZ','OST_WEST_KZ_O', 'D19_LETZTER_KAUF_BRANCHE'], axis=1, inplace=True)

    return df, modes


def cleanNumerical(df, modes, create_modes=False, replaceNanVal=-1):
    for col in df.select_dtypes(['int64']):
        if col == 'RESPONSE':
            continue

        if create_modes:
            modes[col] = df[col].mode().values[0]
        df[col] = df[col].fillna(modes[col])

    for col in df.select_dtypes(['float64']):
        if create_modes:
            modes[col] = df[col].mode().values[0]
        df[col] = df[col].fillna(modes[col])
        df[col] = df[col].astype('int64')

    return df, modes

def cleanData(data_root, input_file_name, output_file_name, attr_val, modes, create_modes, flg_customers=False):
    df = pd.read_csv(os.path.join(data_root, input_file_name), sep=';')
    df = df.set_index(['LNR'])
    if flg_customers:
        df.drop(['CUSTOMER_GROUP', 'ONLINE_PURCHASE', 'PRODUCT_GROUP'], axis=1, inplace=True)
    df = convertUnknowToNan(df, attr_val)  # (191652, 368)
    df = df.drop_duplicates()  # (150103, 368) 21.679398075678835% duplicate data
    df = removeNan(df)
    df, modes = cleanNumerical(df, modes, create_modes)
    df, modes = cleanCategorical(df, modes, create_modes)
    df.to_csv(os.path.join(data_root, output_file_name))
    return df, modes

def loadCleanData(filename_postfix="", flg_azdias=True, flg_customers=True, flg_mailout_train=True, flg_mailout_test=True):
    azdias, customers, mailout_train, mailout_test = None, None, None, None
    
    if flg_azdias:
        azdias = pd.read_csv(os.path.join(data_root, 'azdias_clean'+filename_postfix+'.csv'))
        azdias = azdias.set_index(['LNR'])

    if flg_customers:
        customers = pd.read_csv(os.path.join(data_root, 'customers_clean'+filename_postfix+'.csv'))
        customers = customers.set_index(['LNR'])

    if flg_mailout_train:
        mailout_train = pd.read_csv(os.path.join(data_root, 'mailout_train_clean'+filename_postfix+'.csv'))
        mailout_train = mailout_train.set_index(['LNR'])

    if flg_mailout_test:
        mailout_test = pd.read_csv(os.path.join(data_root, 'mailout_test_clean'+filename_postfix+'.csv'))
        mailout_test = mailout_test.set_index(['LNR'])

    return azdias, customers, mailout_train, mailout_test


if __name__ == '__main__':
    # Data Preparation
    # Attributes Values
    attr_val = pd.read_excel(os.path.join(data_root, '../DIAS Attributes - Values 2017.xlsx'), usecols='B:E', dtype='str').fillna(method='ffill')
    attr_val.columns = attr_val.iloc[0]
    attr_val = attr_val.iloc[1:]
    attr_val = attr_val.drop(['Description'], axis=1)
    attr_val['Attribute'] = attr_val['Attribute'].str.replace('_RZ', '')

    modes = dict()
    _, modes = cleanData(data_root, 'Udacity_AZDIAS_052018.csv', 'azdias_clean.csv', attr_val, modes, True, False)
    _, _ = cleanData(data_root, 'Udacity_CUSTOMERS_052018.csv', 'customers_clean.csv', attr_val, modes, False, True)
    _, _ = cleanData(data_root, 'Udacity_MAILOUT_052018_TRAIN.csv', 'mailout_train_clean.csv', attr_val, modes, False, False)
    _, _ = cleanData(data_root, 'Udacity_MAILOUT_052018_TEST.csv', 'mailout_test_clean.csv', attr_val, modes, False, False)

    # Train
    azdias, customers, mailout_train, _ = loadCleanData('', True, True, True, False)
    df = pd.concat([azdias, customers], axis=0)
    azs, cus = azdias.shape, customers.shape
    del azdias
    del customers

    df['RESPONSE'] = 1
    df.iloc[:azs[0]]['RESPONSE'] = 0
    df = pd.concat([df, mailout_train], axis=0)
    del mailout_train

    X = df.drop(['RESPONSE'], axis=1)
    y = df['RESPONSE']
    del df

    model = Pipeline([
        ('rf', RandomForestClassifier())
    ])
    model.fit(X, y)

    del X
    del y

    # Test
    _, _, _, mailout_test = loadCleanData('', False, False, False, True)
    y_pred = model.predict(mailout_test)
    y_predprob = model.predict_proba(mailout_test)[:, 1]
    result = mailout_test.index.to_frame()
    result['RESPONSE'] = y_predprob
    result.to_csv(os.path.join(data_root, 'result.csv'), index=False)
