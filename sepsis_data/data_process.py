import os
import pandas as pd
import seaborn as sns
import glob
import shutil




mean_value={'HR':80,'O2Sat':90, 'Temp':36.7, 'SBP':120, 'MAP':90, 'DBP':80, 'Resp':18, 'EtCO2':38, 'BaseExcess':0.0,
       'HCO3':25, 'FiO2':35, 'pH':7.40, 'PaCO2':85, 'SaO2':97, 'AST':20, 'BUN':15, 'Alkalinephos':65,
       'Calcium':5, 'Chloride':85, 'Creatinine':1, 'Bilirubin_direct':10, 'Glucose':105,
       'Lactate':220, 'Magnesium':2.5, 'Phosphate':4, 'Potassium':4, 'Bilirubin_total':0.8,
       'TroponinI':0, 'Hct':40, 'Hgb':14, 'PTT':15, 'WBC':6, 'Fibrinogen':300, 'Platelets':200}


def check_type(raw_data_dir,processed_data_dir):
    os.makedirs(processed_data_dir, exist_ok=True)
    paths = glob.glob(raw_data_dir + '/*')
    columns_name = ['FileName', 'TypeSepsis', 'Sex', 'Age', 'Length', 'LenTime']
    list_rows = []
    type = [0, 1]
    for path in paths:
        file_name = path.split('/')[-1].split('.')[0]+'.csv'
        df = pd.read_csv(path, delimiter='|')
        len_time = df.shape[0]
        flag = df[df['SepsisLabel'] == 1].drop_duplicates(subset=['SepsisLabel'])
        age = df['Age'].iloc[0]
        gender = df['Gender'].iloc[0]
        if flag.empty:
            row = [file_name, type[0], gender, age - 1, len_time]
            list_rows.append(row)
        else:
            start = flag.index[0]
            row = [file_name, type[1], gender, age, start, len_time]
            list_rows.append(row)
        print('[***]'+str(path)+' done.')
    save_file = pd.DataFrame(list_rows, columns=columns_name).sort_values(by=['FileName'], ascending=True)
    path_save = processed_data_dir+'/file_info.csv'
    with open(path_save, 'w') as f:
        save_file.to_csv(f, encoding='utf-8', header=True, index=False)
    print('[***] Complete check type.')


def convert_to_csv(raw_data_dir,processed_data_dir):
    os.makedirs(processed_data_dir,exist_ok=True)
    paths = glob.glob(raw_data_dir + '/*')
    for path in paths:
        file_name = path.split('/')[-1].split('.')[0]+'.csv'
        df = pd.read_csv(path, delimiter='|')
        path_save = processed_data_dir + '/' + file_name
        with open(path_save, 'w') as f:
            df[df.columns[:-7]].to_csv(f, encoding='utf-8', header=True, index=False)
        print('[***]' + str(path_save) + ' save.')
    if (os.path.exists(raw_data_dir)):
        shutil.rmtree(raw_data_dir)
    print('[***] Complete convert process.')


def process_missing_data(processed_data_dir,interpolation=True):
    os.makedirs(processed_data_dir, exist_ok=True)
    paths = glob.glob(processed_data_dir + '/*')
    for file in paths:
        df=pd.read_csv(file)
        if interpolation==True:
            df = df.interpolate(method='linear').ffill().bfill()
        df = df.ffill().bfill()
        for c in df.columns:

            if (df[c].isnull().all()):
                df[c] =mean_value[c]

        df.to_csv(file, encoding='utf-8', header=True, index=False)
        print('[***]' + str(file) + ' interpolate done.')
    print('[***] Complete missing data interpolate.')


def concatenate(processed_data_dir):
    set_infor_file=processed_data_dir+'/file_info.csv'
    list_files = pd.read_csv(set_infor_file)
    num_file = list_files.shape[0]
    for i in range(num_file):
        file = os.path.join(processed_data_dir, list_files.iloc[i]['FileName'])
        df = pd.read_csv(file, delimiter='|')
        if i == 0:
            frames = df
        else:
            frames = [frames, df]
            frames = pd.concat(frames)
        os.remove(file)
        print('[***]' + str(file) + ' done.')
    os.remove(set_infor_file)
    path_save = processed_data_dir+'/summary_data.csv'
    with open(path_save, 'w') as f:
        frames.to_csv(f, encoding='utf-8', header=True, index=False)
    print('[*******] Complete summary process.')

def statistical(processed_data_dir):
    file_sum = processed_data_dir+'/summary_data.csv'
    df = pd.read_csv(file_sum)
    name_columns = list(df)
    len = len(name_columns)
    label = name_columns[-1]
    for i in range(len - 1):
        df_temp = df[[name_columns[i], label]]
        df_temp = df_temp.dropna()
        # df_normal = df_temp [df_temp[label]==0].mean()
        # df_sepsis = df_temp [df_temp[label]==1].mean()

        img1 = sns.jointplot(x=name_columns[i], y=label, data=df_temp)
        # img2 = sns.jointplot(x=name_columns[i], y=label, data=df_sepsis)
        # img = df_temp.plot.scatter(x =name_columns[i], y = label, c = 'Red')
        # plt.savefig(img)

        img1.savefig('visualize/' + str(name_columns[i]) + '.png')
        # img2.savefig('./visualize/' + str(name_columns[i]) + '_sepsis.png')

def data_process(raw_data_dir,processed_data_dir):
    check_type(raw_data_dir,processed_data_dir)
    convert_to_csv(raw_data_dir,processed_data_dir)
    process_missing_data(processed_data_dir)

if __name__ == "__main__":

    raw_data_dir = 'training/training_setB'
    processed_data_dir = 'processed_data/training_setB'
    data_process(raw_data_dir,processed_data_dir)