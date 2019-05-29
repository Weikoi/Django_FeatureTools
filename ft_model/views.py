from django.shortcuts import render, render_to_response
from django.http import JsonResponse, HttpResponse
import featuretools
# Create your views here.
from django.template import RequestContext


# 用來接收無效URL的响应
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
    data = ft.demo.load_mock_customer()
    transactions_columns = list(data["transactions"].columns)
    sessions_columns = list(data["sessions"].columns)
    customers_columns = list(data["customers"].columns)
    products_columns = list(data["products"].columns)

    return render(request, "select_tables.html",
                  {"transactions_columns": transactions_columns, "sessions_columns": sessions_columns,
                   "customers_columns": customers_columns, "products_columns": products_columns})


# 用来展示已经选择的表和字段，并且可以选择对应的数据类型
def variables_type(request):
    transactions_columns = request.POST.getlist('transactions')
    sessions_columns = request.POST.getlist('sessions')
    customers_columns = request.POST.getlist('customers')
    products_columns = request.POST.getlist('products')

    response = render(request, "variables_type.html",
                      {"transactions_columns": transactions_columns, "sessions_columns": sessions_columns,
                       "customers_columns": customers_columns, "products_columns": products_columns})
    response.set_cookie('transactions_columns', transactions_columns)
    response.set_cookie('sessions_columns', sessions_columns)
    response.set_cookie('customers_columns', customers_columns)
    response.set_cookie('products_columns', products_columns)

    return response


# 用来展示初始选择的特征和对应的数据类型
def model_parameters(request):
    transactions_types = request.POST.getlist('transactions_types')
    sessions_types = request.POST.getlist('sessions_types')
    customers_types = request.POST.getlist('customers_types')
    products_types = request.POST.getlist('products_types')
    print(customers_types)
    print(products_types)
    transactions_columns = request.COOKIES['transactions_columns']
    sessions_columns = request.COOKIES['sessions_columns']
    customers_columns = request.COOKIES['customers_columns']
    products_columns = request.COOKIES['products_columns']
    print(transactions_columns)
    print(sessions_columns)
    print(customers_columns)
    print(products_columns)
    response = render(request, "model_parameters.html",
                      {"transactions_types": transactions_types, "sessions_types": sessions_types,
                       'customers_types': customers_types, "products_types": products_types,
                       "transactions_columns": transactions_columns, "sessions_columns": sessions_columns,
                       "customers_columns": customers_columns, "products_columns": products_columns, })

    response.set_cookie('transactions_types', transactions_types)
    response.set_cookie('sessions_types', sessions_types)
    response.set_cookie('customers_types', customers_types)
    response.set_cookie('products_types', products_types)

    return response


# 函数selected_features用来处理特征选择提交后服务器响应的结果
def selected_features(request):
    selected = request.POST.getlist('selected')

    columns = list(selected)
    columns.insert(0, 'customer_id')
    import pandas as pd
    df = pd.read_csv("all_features.csv")
    # print(columns)
    new_df = df[columns]
    # print(new_df)
    new_df.to_csv("selected_features.csv", index=False)
    # print(new_df.iloc[0])
    sample_data1 = [round(i, 2) if isinstance(i, float) else i for i in new_df.iloc[0]]
    sample_data2 = [round(i, 2) if isinstance(i, float) else i for i in new_df.iloc[1]]
    sample_data3 = [round(i, 2) if isinstance(i, float) else i for i in new_df.iloc[2]]
    sample_data4 = [round(i, 2) if isinstance(i, float) else i for i in new_df.iloc[3]]
    sample_data5 = [round(i, 2) if isinstance(i, float) else i for i in new_df.iloc[4]]
    print(sample_data1)
    return render(request, "selected_features.html",
                  {"columns": columns, 'sample_data1': sample_data1, 'sample_data2': sample_data2,
                   'sample_data3': sample_data3, 'sample_data4': sample_data4,
                   'sample_data5': sample_data5})


# 函数get_results用来处理模型相关参数提交后服务器响应的结果
def get_results(request):
    print("===================================================================================================")
    import featuretools as ft
    import featuretools
    import pandas as pd
    import numpy as np
    from featuretools.primitives import make_trans_primitive, make_agg_primitive
    # from featuretools.variable_types import *
    # 数据源相关的参数
    transactions_types = eval(request.COOKIES['transactions_types'])
    sessions_types = eval(request.COOKIES['sessions_types'])
    customers_types = eval(request.COOKIES['customers_types'])
    products_types = eval(request.COOKIES['products_types'])

    # 引用规范化，只有在 module 级别可以使用 import *
    transactions_types = ["featuretools.variable_types." + str(i) for i in transactions_types]
    sessions_types = ["featuretools.variable_types." + str(i) for i in sessions_types]
    customers_types = ["featuretools.variable_types." + str(i) for i in customers_types]
    products_types = ["featuretools.variable_types." + str(i) for i in products_types]
    print(transactions_types)
    print(sessions_types)
    print(customers_types)
    print(products_types)

    transactions_columns = eval(request.COOKIES['transactions_columns'])
    sessions_columns = eval(request.COOKIES['sessions_columns'])
    customers_columns = eval(request.COOKIES['customers_columns'])
    products_columns = eval(request.COOKIES['products_columns'])

    print(transactions_columns)
    print(sessions_columns)
    print(customers_columns)
    print(products_columns)

    type_dict1 = {k: eval(v) for k, v in zip(transactions_columns, transactions_types)}
    type_dict2 = {k: eval(v) for k, v in zip(sessions_columns, sessions_types)}
    type_dict3 = {k: eval(v) for k, v in zip(customers_columns, customers_types)}
    type_dict4 = {k: eval(v) for k, v in zip(products_columns, products_types)}

    type_dict = {}
    type_dict.update(type_dict1)
    type_dict.update(type_dict2)
    type_dict.update(type_dict3)
    type_dict.update(type_dict4)
    print("type_dict", type_dict)
    # 模型相关的参数
    max_depth = request.POST['max_depth']
    agg_pri = request.POST.getlist('agg_pri')
    agg_pri_customer = request.POST.getlist('agg_pri_customer')
    trans_pri_customer = request.POST.getlist('trans_pri_customer')
    trans_pri = request.POST.getlist('trans_pri')
    context = {'max_depth': max_depth, 'agg_pri': agg_pri, 'trans_pri': trans_pri}

    pd.set_option('display.max_columns', 20)
    data = ft.demo.load_mock_customer()
    transactions_df = data["transactions"].merge(data["sessions"]).merge(data["customers"]).merge(data["products"])
    print(transactions_df)
    es = ft.EntitySet()

    # 注意type_dict
    es = es.entity_from_dataframe(entity_id="transactions", dataframe=transactions_df, index="transaction_id",
                                  time_index="transaction_time",
                                  variable_types=type_dict)
    sessions_columns.remove("session_id")
    sessions_columns.remove("customer_id")
    # sessions_columns.remove("session_start")
    es = es.normalize_entity(base_entity_id="transactions",
                             new_entity_id="sessions",
                             index="session_id",
                             # make_time_index="session_start",
                             additional_variables=sessions_columns)
    customers_columns.remove("customer_id")
    # customers_columns.remove("join_date")
    es = es.normalize_entity(base_entity_id="transactions",
                             new_entity_id="customers",
                             index="customer_id",
                             # make_time_index="join_date",
                             additional_variables=customers_columns)
    products_columns.remove("product_id")
    es = es.normalize_entity(base_entity_id="transactions",
                             new_entity_id="products",
                             index="product_id",
                             additional_variables=products_columns)

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

    # 将前端页面的提交参数，保存为agg_pri列表
    agg_pri = context['agg_pri']
    trans_pri = context['trans_pri']

    # 如果勾选了参数，加上自定义的Time_since_last_by_hour
    if 'Time_since_last_by_hour' in agg_pri_customer:
        agg_pri.append(Time_since_last_by_hour)
    if 'log_e' in trans_pri_customer:
        trans_pri.append(log)
    # 生成新的特征融合矩阵
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="customers",
                                          agg_primitives=agg_pri,
                                          trans_primitives=trans_pri,
                                          max_depth=int(context['max_depth']))

    # 将索引作为第一列插入数据矩阵
    feature_matrix = feature_matrix.reset_index()
    new_columns = feature_matrix.columns

    # 保存数据矩阵,注意在特征选择界面，没有customer_id作为选项，因为这只是索引
    feature_matrix.to_csv("all_features.csv", index=False)
    res = []
    for i in new_columns:
        res.append(str(i))

    # 将所有的浮点数精度调整到小数点后两位
    sample_data1 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[0]]
    sample_data2 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[1]]
    sample_data3 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[2]]
    sample_data4 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[3]]
    sample_data5 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[4]]

    return render(request, 'get_results.html', {'res': res, 'sample_data1': sample_data1, 'sample_data2': sample_data2,
                                                'sample_data3': sample_data3, 'sample_data4': sample_data4,
                                                'sample_data5': sample_data5})
