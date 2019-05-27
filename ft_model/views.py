from django.shortcuts import render
from django.http import JsonResponse, HttpResponse


# Create your views here.
def index(request):
    return render(request, "index.html")


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

    return render(request, "select_tables.html",
                  {"transactions_columns": transactions_columns, "sessions_columns": sessions_columns,
                   "customers_columns": customers_columns})


def variables_type(request):
    transactions_columns = request.POST.getlist('transactions')
    sessions_columns = request.POST.getlist('sessions')
    customers_columns = request.POST.getlist('customers')

    return render(request, "variables_type.html",
                  {"transactions_columns": transactions_columns, "sessions_columns": sessions_columns,
                   "customers_columns": customers_columns})


# 用來接收無效URL的响应
def no_page(request):
    html = "<h1>There is no page referred to this response</h1>"
    return HttpResponse(html)


# 用来展示初始选择的特征和对应的数据类型
def data_selected(request):
    x = request.POST.getlist('tr')
    y = request.POST.getlist('2')
    y = request.POST.getlist('2')
    print(x)
    print(y)

    return render(request, "data_selected.html",
                  {"transactions_columns": x, "y": y})


# 函数selected_features用来处理 特征选择提交后服务器响应的结果
def selected_features(request):
    selected = request.POST.getlist('selected')

    columns = list(selected)
    columns.insert(0, 'customer_id')
    import pandas as pd
    df = pd.read_csv("all_features.csv")
    print(columns)
    new_df = df[columns]
    print(new_df)
    new_df.to_csv("selected_features.csv", index=False)
    print(new_df.iloc[0])
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


# 函数get_results用来处理表单提交后服务器响应的结果
def get_results(request):
    max_depth = request.POST['max_depth']
    agg_pri = request.POST.getlist('agg_pri')
    agg_pri_customer = request.POST.getlist('agg_pri_customer')
    trans_pri_customer = request.POST.getlist('trans_pri_customer')
    trans_pri = request.POST.getlist('trans_pri')

    context = {'max_depth': max_depth, 'agg_pri': agg_pri, 'trans_pri': trans_pri}

    import featuretools as ft
    import pandas as pd
    import numpy as np
    from featuretools.primitives import make_trans_primitive, make_agg_primitive
    from featuretools.variable_types import DatetimeTimeIndex, Numeric

    pd.set_option('display.max_columns', 20)
    data = ft.demo.load_mock_customer()
    transactions_df = data["transactions"].merge(data["sessions"]).merge(data["customers"])
    products_df = data["products"]

    es = ft.EntitySet()
    s = es.entity_from_dataframe(entity_id="transactions", dataframe=transactions_df, index="transaction_id",
                                 time_index="transaction_time",
                                 variable_types={"product_id": ft.variable_types.Categorical,
                                                 "zip_code": ft.variable_types.ZIPCode})

    es = es.entity_from_dataframe(entity_id="products", dataframe=products_df, index="product_id")

    new_relationship = ft.Relationship(es["products"]["product_id"], es["transactions"]["product_id"])

    es = es.add_relationship(new_relationship)

    es = es.normalize_entity(base_entity_id="transactions",
                             new_entity_id="sessions",
                             index="session_id",
                             make_time_index="session_start",
                             additional_variables=["device", "customer_id", "zip_code", "session_start", "join_date"])

    es = es.normalize_entity(base_entity_id="sessions",
                             new_entity_id="customers",
                             index="customer_id",
                             make_time_index="join_date",
                             additional_variables=["zip_code", "join_date"])

    # feature_matrix1, feature_defs1 = ft.dfs(entityset=es, target_entity="products")
    #
    # feature_matrix2, feature_defs2 = ft.dfs(entityset=es, target_entity="customers", agg_primitives=["count"],
    #                                         trans_primitives=["month"], max_depth=1)

    """
    自定义agg_primitives:
    改写time since last，原函数为秒，现在改为小时输出
    """

    def time_since_last_by_hour(values, time=None):
        time_since = time - values.iloc[-1]
        return time_since.total_seconds() / 3600

    Time_since_last_by_hour = make_agg_primitive(function=time_since_last_by_hour,
                                                 input_types=[DatetimeTimeIndex],
                                                 return_type=Numeric,
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
                               input_types=[Numeric],
                               return_type=Numeric,
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
    feature_matrix3, feature_defs3 = ft.dfs(entityset=es, target_entity="customers",
                                            agg_primitives=agg_pri,
                                            trans_primitives=trans_pri,
                                            max_depth=int(context['max_depth']))

    # 将索引作为第一列插入数据矩阵
    feature_matrix3 = feature_matrix3.reset_index()
    new_columns = feature_matrix3.columns

    # 保存数据矩阵,注意在特征选择界面，没有customer_id作为选项，因为这只是索引
    feature_matrix3.to_csv("all_features.csv", index=False)
    res = []
    for i in new_columns:
        res.append(str(i))

    # 将所有的浮点数精度调整到小数点后两位
    sample_data1 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix3.iloc[0]]
    sample_data2 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix3.iloc[1]]
    sample_data3 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix3.iloc[2]]
    sample_data4 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix3.iloc[3]]
    sample_data5 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix3.iloc[4]]

    return render(request, 'get_results.html', {'res': res, 'sample_data1': sample_data1, 'sample_data2': sample_data2,
                                                'sample_data3': sample_data3, 'sample_data4': sample_data4,
                                                'sample_data5': sample_data5})
