# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser(description='ML Part for the project')
parser.add_argument('--data', type=str, default='../data/data/',
                    help='location of the data')
parser.add_argument('--result', type=str, default='../data/res/',
                    help='location of the result data')
args = parser.parse_args()
args.tied = True

def prepare_matrix(df):
    res_df = df.pivot_table(index='res_id', columns='user_id', values='rating').fillna(0)
    res_df_matrix = csr_matrix(res_df.values)
    return res_df, res_df_matrix

def knn_fit(res_df_matrix):
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(res_df_matrix)
    return model_knn

def get_recommendations(key, res_df, model_knn):
    res = []
    for index, name in enumerate(res_df.index):
        if (str(name) == key) | (name == key):
            test = index
            distance, indices = model_knn.kneighbors(res_df.iloc[test, :].values.reshape(1, -1), n_neighbors=7)
            for i in range(0, len(distance.flatten())):
                if i == 0:
                    # print('Recommendations for {0}:\n'.format(res_df.index[test]))
                    pass
                else:
                    res.append(res_df.index[indices.flatten()[i]])
                    # print('{0}: {1}'.format(i, res_df.index[indices.flatten()[i]]))
            return res
    return res

# topping recommendation since it exists in both algorithms
def merge_res(content_res_lst, user_res_lst):
    intersection_res = set(content_res_lst).intersection(user_res_lst)
    res_lst = list(intersection_res)
    return res_lst

def trans_x(s):
    lst = s.split(" ")
    return [int(i) for i in lst]

def get_search(chicago_data, key_list):
    chicago_data['star'] = chicago_data['features'].apply(lambda x: len(set(trans_x(x)).intersection(key_list)))
    chicago_data = chicago_data.sort_values('star', ascending=False)
    return chicago_data[:6]['id'].tolist()


if __name__ == "__main__":
    features = pd.read_csv(args.data + "features.txt", delimiter='\t', names=['id', 'name'])
    feature_dict = features.set_index('name').T.to_dict('list')
    chicago_data = pd.read_csv(args.data + "chicago.txt", delimiter='\t', names=['id', 'name', 'features'])
    content_based_res = pd.read_csv(args.result + "content_based_chicago.csv")
    name_df = content_based_res[(content_based_res['region'] == 'chicago')][['id','name']]

    demand = input("Enter the restaurant feature you want recommendations for \n")
    demand_list = [feature_dict[i][0] for i in demand.split(", ")]
    # feature_used will be the input for list
    feature_used = demand.split(", ")

    search_lst = get_search(chicago_data, demand_list)
    res_id = search_lst[0]
    print("===========Here is the Search result List==============")
    print(name_df.loc[name_df['id'] == res_id]['name'].values[0])
    train = pd.read_csv(args.result + "session_data_concat.csv")
    res_df, res_df_matrix = prepare_matrix(train)
    model_knn = knn_fit(res_df_matrix)
    user_res = get_recommendations(res_id, res_df, model_knn)

    user_res_lst = []
    for i in user_res:
        user_res_lst.append(name_df.loc[name_df['id'] == i]['name'].values[0])

    content_df = content_based_res[(content_based_res['id'] == int(res_id)) & (content_based_res['region'] == 'chicago')]
    
    name_res1 = name_df.loc[name_df['id'] == content_df['res1'].values[0]]['name'].values[0]
    name_res2 = name_df.loc[name_df['id'] == content_df['res2'].values[0]]['name'].values[0]
    name_res3 = name_df.loc[name_df['id'] == content_df['res3'].values[0]]['name'].values[0]
    name_res4 = name_df.loc[name_df['id'] == content_df['res4'].values[0]]['name'].values[0]
    name_res5 = name_df.loc[name_df['id'] == content_df['res5'].values[0]]['name'].values[0]
    name_res6 = name_df.loc[name_df['id'] == content_df['res6'].values[0]]['name'].values[0]

    content_res_lst = [name_res1,name_res2,name_res3,name_res4,name_res5,name_res6]
    merge_res_lst = merge_res(content_res_lst, user_res_lst)
    print("===========Here is the User-Based recommendation List==============")
    print(user_res_lst)
    print("===========Here is the Content-Based recommendation List==============")
    print(content_res_lst)
    print("===========Here is the Intersection recommendation List of above two methods ==============")
    print(merge_res_lst)
