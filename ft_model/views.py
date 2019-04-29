from django.shortcuts import render
from django.http import JsonResponse, HttpResponse

# Create your views here.
from featuretools.primitives import make_agg_primitive


def index(request):
    return render(request, "index2.html")


def no_page(request):
    html = "<h1>There is no page referred to this response</h1>"
    return HttpResponse(html)


# 函数postTest2用来处理表单提交后服务器响应的结果
def get_results(request):
    max_depth = request.POST['max_depth']
    agg_pri = request.POST.getlist('agg_pri')
    trans_pri = request.POST.getlist('trans_pri')

    context = {'max_depth': max_depth, 'agg_pri': agg_pri, 'trans_pri': trans_pri}

    import featuretools as ft
    import pandas as pd
    from featuretools.primitives import make_trans_primitive
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
        return time_since.total_seconds()/3600
        # return max(column)

    Time_since_last_by_hour = make_agg_primitive(function=time_since_last_by_hour,
                                                 input_types=[DatetimeTimeIndex],
                                                 return_type=Numeric,
                                                 uses_calc_time=True)

    # 将前端页面的提交参数，保存为agg_pri列表
    agg_pri = context['agg_pri']

    # 加上自定义的Time_since_last_by_hour
    agg_pri.append(Time_since_last_by_hour)
    feature_matrix3, feature_defs3 = ft.dfs(entityset=es, target_entity="customers",
                                            agg_primitives=agg_pri,
                                            trans_primitives=context['trans_pri'],
                                            max_depth=int(context['max_depth']))
    res = []
    for i in feature_defs3:
        res.append(str(i))

    sample_data = [i for i in feature_matrix3.iloc[0]]
    return render(request, 'get_results.html', {'res': res, 'sample_data': sample_data})
