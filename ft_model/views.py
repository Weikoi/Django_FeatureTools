from django.shortcuts import render, render_to_response
from django.http import JsonResponse, HttpResponse
import featuretools
from django.template import RequestContext


# Create your views here.


# 用來接收无效URL的响应
def no_page(request):
    html = "<h1>There is no page referred to this response</h1>"
    return HttpResponse(html)


# 用来展现对于表和字段的选择
def select_tables(request):
    import featuretools as ft
    import pandas as pd
    import numpy as np
    from featuretools.primitives import make_trans_primitive, make_agg_primitive
    from featuretools.variable_types import DatetimeTimeIndex, Numeric
    import os
    import re

    # 将接口改成对应CSV的api
    if not os.path.isdir(os.getcwd() + "/demo_data"):
        os.mkdir(os.getcwd() + "/demo_data")
    os.chdir(os.getcwd() + "/demo_data")
    print(os.getcwd())
    print(os.listdir(os.getcwd()))
    regex = re.compile("csv")
    raw_dict = {}
    for file in os.listdir(os.getcwd()):
        if re.search(regex, file):
            raw_dict[file.split(".")[0]] = pd.read_csv(file)
    data = raw_dict
    os.chdir("..")

    columns_list = []
    name_list = []

    for k, v in data.items():
        columns_list.append(list(v.columns))
        name_list.append(k)
    columns_dict = {k: v for k, v in zip(name_list, columns_list)}
    response = render(request, "select_tables.html",
                      {"columns_dict": columns_dict})
    response.set_cookie('columns_dict', columns_dict)
    return response


# 用来展示已经选择的表和字段，并且可以选择对应的数据类型
def variables_type(request):
    columns_dict = request.COOKIES['columns_dict']
    # print(columns_dict)
    # print("======================")
    return render(request, "variables_type.html",
                  {"columns_dict": eval(columns_dict)})


# 用来展示初始选择的特征和对应的数据类型
def model_parameters(request):
    # print(customers_types)
    # print(products_types)
    types_list = []
    name_list = []

    # target 是融合的对象
    columns_dict = request.COOKIES['columns_dict']
    target = request.POST.get("target")
    # print("++++++")
    # print(target)
    # print("++++++")
    # print("++++++")
    # print(columns_dict)
    # print("++++++")
    for k, v in eval(columns_dict).items():
        types_list.append(request.POST.getlist(k))
        name_list.append(k)
    types_dict = {k: v for k, v in zip(name_list, types_list)}

    response = render(request, "model_parameters.html",
                      {"types_dict": types_dict,
                       "columns_dict": columns_dict, })
    # print(types_dict)
    # target_id = ''
    # for type_i, column_i in zip(types_dict[target], columns_dict[target]):
    #     if 'Index' in type_i:
    #         target_id = column_i
    # print("=============")
    # print(target_id)
    # print("=============")
    response.set_cookie('types_dict', types_dict)
    # response.set_cookie('target_id', target_id)
    response.set_cookie('target', target)
    return response


# 函数get_results用来处理模型相关参数提交后服务器响应的结果
def get_results(request):
    try:
        import featuretools as ft
        import pandas as pd
        import numpy as np
        from featuretools.primitives import make_trans_primitive, make_agg_primitive

        # 数据源相关的参数
        types_dict = eval(request.COOKIES['types_dict'])
        columns_dict = eval(request.COOKIES['columns_dict'])
        target = request.COOKIES['target']

        # 如何决定 base entity?
        # 目前思路是由 id 类型最多的 entity 来做 base entity
        # 把对应的表和id个数封装成字典，然后根据个数给表名排逆序，然后按照这个顺序merge表，是为最终思路
        base_entity = ''
        base_index = ''

        max_count = 0
        sorted_dict = {}
        for k, v in types_dict.items():
            count = 0

            index = ''
            for i in v:
                if '.Id' in str(i):
                    count += 1
                if '.Index' in str(i):
                    index = i
            sorted_dict[k] = count
            if count > max_count:
                base_entity = k
                base_index = index
                max_count = count
        sorted_list = sorted(sorted_dict.items(), key=lambda item: item[1], reverse=True)
        sorted_table_name = [i[0] for i in sorted_list]

        print("sorted_table_name\n", sorted_table_name)

        # 把columns 和对应的 类型拼接成字典，存在一个列表中,并且找到base_index
        types_dict_list = []
        entity_name_list = []
        for key, values1, values2 in zip(columns_dict.keys(), columns_dict.values(), types_dict.values()):
            types_dict_list.append({k: eval(v) for k, v in zip(values1, values2)})
            entity_name_list.append(key)
            if key == base_entity:
                for k, v in zip(values2, values1):
                    if '.Index' in k:
                        base_index = v

        # 自动识别标记为Index的特征，并作为抽取实体的index参数，传入模型
        # 把所有的类型字典拼成一个大字典
        index_list = []
        total_type_dict = {}
        for each_dict in types_dict_list:
            total_type_dict.update(each_dict)
            for k, v in each_dict.items():
                if '.Index' in str(v):
                    index_list.append(k)
        print(index_list)
        # print(total_type_dict)

        # 原表全部join在一起之后再抽取实体
        # 数据接口改成处理CSV结构
        import os
        import re
        if not os.path.isdir(os.getcwd() + "/demo_data"):
            os.mkdir(os.getcwd() + "/demo_data")
        os.chdir(os.getcwd() + "/demo_data")
        regex = re.compile("csv")
        raw_dict = {}

        for file in os.listdir(os.getcwd()):
            if re.search(regex, file):
                raw_dict[file.split(".")[0]] = pd.read_csv(file)

        data = raw_dict
        os.chdir("..")

        # todo : merge的逻辑比较复杂，要如何执行join操作？？
        if len(data) == 0:
            raise Exception("数据源为空，请检查数据源文件")
        elif len(data) > 1:
            data_df = data.pop(sorted_table_name.pop(0))
            # print(data_df)
            for i in sorted_table_name:
                data_df = data_df.merge(data[i])
            #
            # for i in list(data.values()):
            #     data_df = data_df.merge(i)

        elif len(data) == 1:
            data_df = list(data.values())[0]
        es = ft.EntitySet()

        # print("+++++++++++++++++++++++")
        # print("data_df\n", data_df)
        # print("entity_id\n", base_entity)
        # print("base_index\n", base_index)
        # print("total_type_dict\n", total_type_dict)
        # print("+++++++++++++++++++++++")
        # 构造base entity, 将第一个表名作为基础实体名称
        es = es.entity_from_dataframe(entity_id=base_entity,
                                      dataframe=data_df,
                                      index=base_index,
                                      # time_index="transaction_time",
                                      variable_types=total_type_dict)

        # 基于base entity抽取实体,逻辑比较复杂，基本逻辑是作为base entity的字段，跳过实体抽取，其余的将index 字段单独存储，设为index参数
        for k, v in columns_dict.items():
            if k == base_entity:
                continue
            index = ''
            for i in index_list:
                if i in v:
                    v.remove(i)
                    index = i
            # print("=========")
            # print(k)
            # print(index)
            # print(v)
            # print("=========")
            es = es.normalize_entity(base_entity_id=base_entity,
                                     new_entity_id=k,
                                     index=index,
                                     # make_time_index="session_start",
                                     additional_variables=v)

        """
        自定义agg_primitives:
        改写time since last，原函数为秒，现在改为小时输出
        """

        def time_since_last_by_hour(values, time=None):
            time_since = time - values.iloc[-1]
            return time_since.total_seconds() / 3600

        Time_since_last_by_hour = make_agg_primitive(function=time_since_last_by_hour,
                                                     input_types=[ft.variable_types.DatetimeTimeIndex],
                                                     return_type=ft.variable_types.Numeric,
                                                     uses_calc_time=True)

        """
        自定义trans_primitives:
        添加log e 的自然对数
        """
        import numpy as np

        def log(vals):
            return np.log(vals)

        # def generate_name(self, base_feature_names):
        #     return "-(%s)" % (base_feature_names[0])
        log = make_trans_primitive(function=log,
                                   input_types=[ft.variable_types.Numeric],
                                   return_type=ft.variable_types.Numeric,
                                   # uses_calc_time=True,
                                   description="Calculates the log of the value.",
                                   name="log")

        # 模型相关的参数
        max_depth = request.POST['max_depth']
        agg_pri = request.POST.getlist('agg_pri')
        agg_pri_customer = request.POST.getlist('agg_pri_customer')
        trans_pri_customer = request.POST.getlist('trans_pri_customer')
        trans_pri = request.POST.getlist('trans_pri')
        context = {'max_depth': max_depth, 'agg_pri': agg_pri, 'trans_pri': trans_pri}

        pd.set_option('display.max_columns', 20)

        # 将前端页面的提交参数，保存为agg_pri列表
        agg_pri = context['agg_pri']
        trans_pri = context['trans_pri']

        # 如果勾选了参数，加上自定义的Time_since_last_by_hour
        if 'Time_since_last_by_hour' in agg_pri_customer:
            agg_pri.append(Time_since_last_by_hour)
        if 'log_e' in trans_pri_customer:
            trans_pri.append(log)

        # 生成新的特征融合矩阵
        feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity=target,
                                              agg_primitives=agg_pri,
                                              trans_primitives=trans_pri,
                                              max_depth=int(context['max_depth']))

        # 将索引作为第一列插入数据矩阵
        feature_matrix = feature_matrix.reset_index()
        new_columns = feature_matrix.columns

        # 保存数据矩阵,注意在特征选择界面，没有 customer_id 作为选项，因为这只是索引
        # nlp 数组是将primitives替换为中文后的表头，一并显示在第二行
        import os
        if not os.path.isdir(os.getcwd() + "/demo_data/result"):
            os.mkdir(os.getcwd() + "/demo_data/result")
        feature_matrix.to_csv("./demo_data/result/all_features.csv", index=False)
        # print(feature_matrix.head(5))
        from .columns2NLP import columns2NLP
        res = []
        nlp = []
        for i in new_columns:
            res.append(str(i))
            nlp.append(columns2NLP(str(i)))
        # print(res[0])
        # print("======================")
        # print(res)
        # print(nlp)
        # print("======================")
        # 将所有的浮点数精度调整到小数点后两位
        sample_data1 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[0]]
        sample_data2 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[1]]
        sample_data3 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[2]]
        sample_data4 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[3]]
        sample_data5 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[4]]
        response = render(request, 'get_results.html', {'res': res,
                                                        'nlp': nlp,
                                                        'sample_data1': sample_data1,
                                                        'sample_data2': sample_data2,
                                                        'sample_data3': sample_data3,
                                                        'sample_data4': sample_data4,
                                                        'sample_data5': sample_data5})
        response.set_cookie('target_id', res[0])
        return response

    except Exception as e:
        response = render(request, 'erro.html', {'erro': e})
        return response


# 函数selected_features用来处理特征选择提交后服务器响应的结果
def selected_features(request):
    import re
    selected = request.POST.getlist('selected')
    target_id = request.COOKIES['target_id']
    columns = list(selected)
    columns.insert(0, target_id)
    import pandas as pd
    df = pd.read_csv("./demo_data/result/all_features.csv")
    # print(columns)
    new_df = df[columns]
    # print(new_df)
    new_df.to_csv("./demo_data/result/selected_features.csv", index=False)

    # print(new_df.iloc[0])

    # 显示的时候由于pandas中全部统一处理成float，导致ID之类的整形数变成带小数点的，目前没有找到更好的解决办法，
    # 只好使用正则表达式进行区分打印，注意，这只是打印，与存储无关。

    # print("+_++__+_+_+_+_+_+_+_+_+_")
    # print(new_df.iloc[0])
    # print("+_++__+_+_+_+_+_+_+_+_+_")

    def transfer(data_sample):
        return_list = []
        for i in data_sample:
            if i is None:
                return_list.append(None)
            if isinstance(i, str):
                return_list.append(i)
            elif re.search(r"\.0\b", str(i)):
                return_list.append(int(i))
            else:
                return_list.append(round(i, 2))
        return return_list

    sample_data1 = transfer(new_df.iloc[0])
    sample_data2 = transfer(new_df.iloc[1])
    sample_data3 = transfer(new_df.iloc[2])
    sample_data4 = transfer(new_df.iloc[3])
    sample_data5 = transfer(new_df.iloc[4])

    # print(sample_data1)
    return render(request, "selected_features.html",
                  {"columns": columns, 'sample_data1': sample_data1, 'sample_data2': sample_data2,
                   'sample_data3': sample_data3, 'sample_data4': sample_data4,
                   'sample_data5': sample_data5})
