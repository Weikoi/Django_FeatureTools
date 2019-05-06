# %%
import featuretools as ft
import pandas as pd
import numpy as np
from featuretools.primitives import make_trans_primitive, make_agg_primitive
from featuretools.variable_types import DatetimeTimeIndex, Numeric

# sessions 是客户打开一次客户端的行为
# 而transactions是他的交易信息，一次打开客户端可能会完成多笔交易。
# %%
pd.set_option('display.max_columns', 20)
data = ft.demo.load_mock_customer()
transactions_df = data["transactions"].merge(data["sessions"]).merge(data["customers"])
products_df = data["products"]

# %%
es = ft.EntitySet()
es = es.entity_from_dataframe(entity_id="transactions", dataframe=transactions_df, index="transaction_id",
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

"""
自定义max2:
求values第二大的数
"""


def max2nd(vals):
    return sorted(vals)[-2]


max2nd = make_agg_primitive(function=max2nd,
                            input_types=[Numeric],
                            return_type=Numeric,
                            # uses_calc_time=True,
                            description="Calculates the second max of the value.",
                            name="max2nd")

"""
自定义max3:取第三大的数
"""


def max3rd(vals):
    return sorted(vals)[-2]


max3rd = make_agg_primitive(function=max3rd,
                            input_types=[Numeric],
                            return_type=Numeric,
                            # uses_calc_time=True,
                            description="Calculates the second max of the value.",
                            name="max3rd")

# 生成新的特征融合矩阵
feature_matrix3, feature_defs3 = ft.dfs(entityset=es, target_entity="customers",
                                        agg_primitives=[max3rd, max2nd],
                                        trans_primitives=[log],
                                        max_depth=2)

print(feature_defs3)
