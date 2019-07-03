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
pd.set_option('display.width', 1000)
data = ft.demo.load_mock_customer()

# %%
transactions_df = data["transactions"].merge(data["sessions"]).merge(data["customers"]).merge(data["products"])

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
                         make_time_index="session_start",
                         additional_variables=["device", "customer_id", "zip_code", "session_start", "join_date"])

es = es.normalize_entity(base_entity_id="sessions",
                         new_entity_id="customers",
                         index="customer_id",
                         make_time_index="join_date",
                         additional_variables=["zip_code", "join_date"])

es = es.normalize_entity(base_entity_id="transactions",
                         new_entity_id="products",
                         index="product_id",
                         additional_variables=["brand"])
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

"""
自定义trans_primitives:
判断是否为正数
"""
import numpy as np


def is_positive(vals):
    return vals > 0


# def generate_name(self, base_feature_names):
#     return "-(%s)" % (base_feature_names[0])
is_positive = make_trans_primitive(function=is_positive,
                                   input_types=[ft.variable_types.Numeric],
                                   return_type=ft.variable_types.Boolean,
                                   # uses_calc_time=True,
                                   description="Calculates if the value positive.",
                                   name="is_positive")


feature_matrix, feature_defs = ft.dfs(entityset=es,
                                      target_entity="customers",
                                        agg_primitives=["max"],
                                      trans_primitives=["month", is_positive],
                                      max_depth=2)

print(feature_matrix)

# es = es.normalize_entity(base_entity_id="transactions",
#                          new_entity_id="customers",
#                          index="customer_id",
#                          # make_time_index="join_date",
#                          additional_variables=["zip_code", "join_date"])
#
# es = es.normalize_entity(base_entity_id="transactions",
#                          new_entity_id="products",
#                          index="product_id",
#                          additional_variables=["brand"])
#
# # feature_matrix1, feature_defs1 = ft.dfs(entityset=es, target_entity="products")
# #
# # feature_matrix2, feature_defs2 = ft.dfs(entityset=es, target_entity="customers", agg_primitives=["count"],
# #                                         trans_primitives=["month"], max_depth=1)
#
# print(es)
#
#
# def time_since_last_by_hour(values, time=None):
#     time_since = time - values.iloc[-1]
#     return time_since.total_seconds() / 3600
#
#
# Time_since_last_by_hour = make_agg_primitive(function=time_since_last_by_hour,
#                                              input_types=[ft.variable_types.DatetimeTimeIndex],
#                                              return_type=ft.variable_types.Numeric,
#                                              uses_calc_time=True)
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
# """
# 自定义trans_primitives:
# 判断是否为正数
# """
# import numpy as np
#
#
# def is_positive(vals):
#     return vals > 0
#
#
# # def generate_name(self, base_feature_names):
# #     return "-(%s)" % (base_feature_names[0])
# is_positive = make_trans_primitive(function=is_positive,
#                                    input_types=[ft.variable_types.Numeric],
#                                    return_type=ft.variable_types.Boolean,
#                                    # uses_calc_time=True,
#                                    description="Calculates if the value positive.",
#                                    name="is_positive")
# # """
# # # 生成新的特征融合矩阵
# # # 可以根据target_entity的不同生成不同的融合特征矩阵
# # """
# feature_matrix, feature_defs = ft.dfs(entityset=es,
#                                       target_entity="customers",
#                                       #   agg_primitives=["median", "count", "num_unique", "max","avg_time_between", "n_most_common", max2nd, max3rd],
#                                       agg_primitives=["count", "mean", "skew", "time_since_last_by_hour"],
#                                       trans_primitives=["time_since", "log"],
#                                       max_depth=2)
#
# # %%
# print(feature_defs)
#
# # %%
# print(feature_matrix)
