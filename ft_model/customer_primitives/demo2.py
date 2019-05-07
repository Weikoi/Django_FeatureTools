# %%
import featuretools as ft
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
data = ft.demo.load_mock_customer()
transactions_df = data["transactions"].merge(
    data["sessions"]).merge(data["customers"]).merge(data["products"])

# %%
es = ft.EntitySet()
es = es.entity_from_dataframe(entity_id="transactions", dataframe=transactions_df, index="transaction_id",
                              time_index="transaction_time",
                              variable_types={"product_id": ft.variable_types.Categorical,
                                              "zip_code": ft.variable_types.ZIPCode})

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

# feature_matrix1, feature_defs1 = ft.dfs(entityset=es, target_entity="products")
#
# feature_matrix2, feature_defs2 = ft.dfs(entityset=es, target_entity="customers", agg_primitives=["count"],
#                                         trans_primitives=["month"], max_depth=1)


# %%
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
参数name为生成后的特征所显示的名字，而接受make_transprimitive的参数为传给dfs的基元名，注意是变量，而不是字符串（自定义的特殊）
"""


def log(vals):
    return np.log(vals)


log = make_trans_primitive(function=log,
                           input_types=[Numeric],
                           return_type=Numeric,
                           # uses_calc_time=True,
                           description="Calculates the log of the value.",
                           name="log_e")

"""
自定义max2:取第二大的数
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

# %%


"""
# 生成新的特征融合矩阵
# 可以根据target_entity的不同生成不同的融合特征矩阵
"""
feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="products",
                                      agg_primitives=["median", "count", "num_unique", "max","avg_time_between", max2nd, max3rd],
                                      trans_primitives=["month"],
                                      max_depth=3)

# %%
feature_defs

# %%
feature_matrix


# %%
"""
# 注意target_entity与cutoff_times的时间戳是对应的，如果target是客户，那么设立的应当是customer_id,如果target是product,那么设立的应当是product_id
# 所以注意二者之间的匹配关系
"""
cutoff_times = pd.DataFrame()

cutoff_times['product_id'] = [1, 2, 3, 1]

cutoff_times['time'] = pd.to_datetime(['2014-1-1 04:00',
                                       '2014-1-1 05:00',
                                       '2014-1-1 06:00',
                                       '2014-1-1 08:00'])
# cutoff_times['label'] = [True, True, False, True]

# %%
cutoff_times

# %%
fm, features = ft.dfs(entityset=es,
                      target_entity='products',
                      cutoff_time=cutoff_times,
                      cutoff_time_in_index=True)

# %%
fm
# %%
features


# %%
"""
# 设立时间窗口必须先声明添加 last_time_index
# 其次cutoff为特征计算的截止时间，时间窗口是在这截止时间之前的窗口！！！
# training_window可以有的单位：second, minute,hour, day, week, year, 注意没有month, 这是源码中check_timedelta的api所决定的
"""
es.add_last_time_indexes()
window_fm, window_features = ft.dfs(entityset=es,
                                    target_entity="customers",
                                    cutoff_time=cutoff_times,
                                    cutoff_time_in_index=True,
                                    training_window="1 hour")

# %%
window_fm
# %%
window_features

# %%
es['sessions'].df['session_start'].head()

# %%
es['sessions'].last_time_index.head()
