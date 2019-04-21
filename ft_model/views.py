from django.shortcuts import render
from django.http import JsonResponse


# Create your views here.
def index(request):
    return render(request, "index.html")


# 函数postTest2用来处理表单提交后服务器响应的结果
def get_results(request):
    max_depth = request.POST['depth']
    agg_pri = request.POST.getlist('agg_pri')
    trans_pri = request.POST.getlist('trans_pri')

    context = {'max_depth': max_depth, 'agg_pri': agg_pri, 'trans_pri': trans_pri}

    import featuretools as ft
    import pandas as pd

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

    feature_matrix1, feature_defs1 = ft.dfs(entityset=es, target_entity="products")

    feature_matrix2, feature_defs2 = ft.dfs(entityset=es, target_entity="customers", agg_primitives=["count"],
                                            trans_primitives=["month"], max_depth=1)

    feature_matrix3, feature_defs3 = ft.dfs(entityset=es, target_entity="customers",
                                            agg_primitives=context['agg_pri'],
                                            trans_primitives=context['trans_pri'],
                                            max_depth=int(context['max_depth']))
    res = []

    for i in feature_defs3:
        res.append(str(i))

    return render(request, 'get_results.html', {'res': res})
