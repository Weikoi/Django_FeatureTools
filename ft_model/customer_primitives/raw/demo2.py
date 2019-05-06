from featuretools.primitives import AggregationPrimitive, DatetimeTimeIndex, Numeric, make_trans_primitive, \
    make_agg_primitive
import featuretools as ft
import pandas as pd

pd.set_option('display.max_columns', 20)

es = ft.demo.load_mock_customer(return_entityset=True)


class TimeSinceLast(AggregationPrimitive):
    """Time since last related instance."""
    name = "time_since_last"
    input_types = [DatetimeTimeIndex]
    return_type = Numeric
    uses_calc_time = True

    def get_function(self):
        def time_since_last(values, time=None):
            time_since = time - values.iloc[-1]
            return time_since.total_seconds()

        return time_since_last


def time_since_last_by_hour(values, time=None):
    # print(type(values))
    # print(values.iloc[-1])
    time_since = time - values.iloc[-1]
    return time_since.total_seconds() / 3600
    # return max(column)


Time_since_last_by_hour = make_agg_primitive(function=time_since_last_by_hour,
                                             input_types=[DatetimeTimeIndex],
                                             return_type=Numeric,
                                             uses_calc_time=True)

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

feature_matrix3, feature_defs3 = ft.dfs(entityset=es, target_entity="customers",
                                        # agg_primitives=[Time_since_last_by_hour],
                                        trans_primitives=[log],
                                        max_depth=int(2))

print(feature_matrix3)
