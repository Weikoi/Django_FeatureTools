# %%
import featuretools as ft
import featuretools
import pandas as pd
import numpy as np
from featuretools.primitives import make_trans_primitive, make_agg_primitive
from featuretools.variable_types import DatetimeTimeIndex, Numeric

"""
# sessions 是客户打开一次客户端的行为
# 而transactions是他的交易信息，一次打开客户端可能会完成多笔交易。
"""
# %%
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)
data = ft.demo.load_mock_customer()

# %%
transactions_df = data["transactions"].merge(
    data["sessions"])

# %%
es = ft.EntitySet()
es = es.entity_from_dataframe(entity_id="transactions",
                              dataframe=transactions_df,
                              index="transaction_id",
                              # time_index="transaction_time",
                              variable_types={'transaction_id': featuretools.variable_types.variable.Index,
                                              'session_id': featuretools.variable_types.variable.Index,
                                              'transaction_time': featuretools.variable_types.variable.TimeIndex,
                                              'product_id': featuretools.variable_types.variable.Id,
                                              'amount': featuretools.variable_types.variable.Numeric,
                                              'customer_id': featuretools.variable_types.variable.Id,
                                              'device': featuretools.variable_types.variable.Categorical,
                                              'session_start': featuretools.variable_types.variable.TimeIndex})

es = es.normalize_entity(base_entity_id="transactions",
                         new_entity_id="sessions",
                         index="session_id",
                         # make_time_index="session_start",
                         additional_variables=["device", "customer_id", "session_start"])

# es = es.normalize_entity(base_entity_id="sessions",
#                          new_entity_id="customers",
#                          index="customer_id",
#                          # make_time_index="join_date",
#                          additional_variables=["zip_code", "join_date"])
#
# es = es.normalize_entity(base_entity_id="transactions",
#                          new_entity_id="products",
#                          index="product_id",
#                          additional_variables=["brand"])

# feature_matrix1, feature_defs1 = ft.dfs(entityset=es, target_entity="products")
#
# feature_matrix2, feature_defs2 = ft.dfs(entityset=es, target_entity="customers", agg_primitives=["count"],
#                                         trans_primitives=["month"], max_depth=1)


print(es)
# """
# # 生成新的特征融合矩阵
# # 可以根据target_entity的不同生成不同的融合特征矩阵
# """
feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="sessions",
                                      #   agg_primitives=["median", "count", "num_unique", "max","avg_time_between", "n_most_common", max2nd, max3rd],
                                      agg_primitives=["count", "mean", "skew"],
                                      trans_primitives=["time_since", "is_weekend", "weekday", "time_since_previous"],
                                      max_depth=2)

# %%
print(feature_defs)

# %%
print(feature_matrix)
