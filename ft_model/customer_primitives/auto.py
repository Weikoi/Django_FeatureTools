#%% md
aggregation primitive
#%%
from featuretools.variable_types import (Index, Numeric, Discrete, Boolean,
                                         DatetimeTimeIndex, Variable)
from .aggregation_primitive_base import (AggregationPrimitive,
                                         make_agg_primitive)
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import skew


# TODO: make sure get func gets numpy arrays not series


class Count(AggregationPrimitive):
    """Counts the number of non null values"""
    name = "count"
    input_types = [[Index], [Variable]]
    return_type = Numeric
    stack_on_self = False
    default_value = 0

    def __init__(self, id_feature, parent_entity, count_null=False, **kwargs):
        self.count_null = count_null
        super(Count, self).__init__(id_feature, parent_entity, **kwargs)

    def get_function(self):
        def func(values, count_null=self.count_null):
            if len(values) == 0:
                return 0

            if count_null:
                values = values.fillna(0)

            return values.count()
        return func

    def _get_name(self):
        where_str = self._where_str()
        use_prev_str = self._use_prev_str()

        return u"COUNT(%s%s%s)" % (self.child_entity.name,
                                   where_str, use_prev_str)


class Sum(AggregationPrimitive):
    """Counts the number of elements of a numeric or boolean feature"""
    name = "sum"
    input_types = [Numeric]
    return_type = Numeric
    stack_on_self = False
    stack_on_exclude = [Count]

    # todo: handle count nulls
    def get_function(self):
        def sum_func(x):
            return np.nan_to_num(x.values).sum(dtype=np.float)
        return sum_func


class Mean(AggregationPrimitive):
    """Computes the average value of a numeric feature"""
    name = "mean"
    input_types = [Numeric]
    return_type = Numeric

    # p todo: handle nulls
    def get_function(self):
        return np.nanmean


class Mode(AggregationPrimitive):
    """Finds the most common element in a categorical feature"""
    name = "mode"
    input_types = [Discrete]
    return_type = None

    def get_function(self):
        def pd_mode(x):
            if x.mode().shape[0] == 0:
                return np.nan
            return x.mode().iloc[0]
        return pd_mode


Min = make_agg_primitive(
    np.min,
    [Numeric],
    None,
    name="min",
    stack_on_self=False,
    description="Finds the minimum non-null value of a numeric feature.")


# class Min(AggregationPrimitive):
#     """Finds the minimum non-null value of a numeric feature."""
#     name = "min"
#     input_types =  [Numeric]
#     return_type = None
#     # max_stack_depth = 1
#     stack_on_self = False

#     def get_function(self):
#         return np.min


class Max(AggregationPrimitive):
    """Finds the maximum non-null value of a numeric feature"""
    name = "max"
    input_types = [Numeric]
    return_type = None
    # max_stack_depth = 1
    stack_on_self = False

    def get_function(self):
        return np.max


class NUnique(AggregationPrimitive):
    """Returns the number of unique categorical variables"""
    name = "num_unique"
    # todo can we use discrete in input_types instead?
    input_types = [Discrete]
    return_type = Numeric
    # max_stack_depth = 1
    stack_on_self = False

    def get_function(self):
        return lambda x: x.nunique()


class NumTrue(AggregationPrimitive):
    """Finds the number of 'True' values in a boolean"""
    name = "num_true"
    input_types = [Boolean]
    return_type = Numeric
    default_value = 0
    stack_on = []
    stack_on_exclude = []

    def get_function(self):
        def num_true(x):
            return np.nan_to_num(x.values).sum()
        return num_true


class PercentTrue(AggregationPrimitive):
    """Finds the percent of 'True' values in a boolean feature"""
    name = "percent_true"
    input_types = [Boolean]
    return_type = Numeric
    max_stack_depth = 1
    stack_on = []
    stack_on_exclude = []

    def get_function(self):
        def percent_true(x):
            if len(x) == 0:
                return np.nan
            return np.nan_to_num(x.values).sum(dtype=np.float) / len(x)
        return percent_true

'''
函数应用举例：
最喜爱的top_n 产品/股票/行业/……
'''
class NMostCommon(AggregationPrimitive):
    """Finds the N most common elements in a categorical feature"""
    name = "n_most_common"
    input_types = [Discrete]
    return_type = Discrete
    # max_stack_depth = 1
    stack_on = []
    stack_on_exclude = []
    expanding = True

    def __init__(self, base_feature, parent_entity, n=3):
        self.n = n
        super(NMostCommon, self).__init__(base_feature, parent_entity)

    @property
    def default_value(self):
        return np.zeros(self.n) * np.nan

    def get_expanded_names(self):
        names = []
        for i in range(1, self.n + 1):
            names.append(str(i) + self.get_name()[1:])
        return names

    def get_function(self):
        def pd_topn(x, n=self.n):
            return np.array(x.value_counts()[:n].index)
        return pd_topn

'''
函数应用举例：
客户平均登录时长/平均购买**产品间隔/……
'''
class AvgTimeBetween(AggregationPrimitive):
    """Computes the average time between consecutive events
    using the time index of the entity.
    Note: equivalent to Mean(Diff(time_index)), but more performant
    """

    # Potentially unnecessary if we add an trans_feat that
    # calculates the difference between events. DFS
    # should then calculate the average of that trans_feat
    # which amounts to AvgTimeBetween
    name = "avg_time_between"
    input_types = [DatetimeTimeIndex]
    return_type = Numeric
    # max_stack_depth = 1

    def get_function(self):
        def pd_avg_time_between(x):
            """
            Assumes time scales are closer to order
            of seconds than to nanoseconds
            if times are much closer to nanoseconds
            we could get some floating point errors
            this can be fixed with another function
            that calculates the mean before converting
            to seconds
            """
            x = x.dropna()
            if x.shape[0] < 2:
                return np.nan
            if isinstance(x.iloc[0], (pd.Timestamp, datetime)):
                x = x.astype('int64')
                # use len(x)-1 because we care about difference
                # between values, len(x)-1 = len(diff(x))
                avg = ((x.max() - x.min())) / float(len(x) - 1)
            else:
                avg = (x.max() - x.min()) / float(len(x) - 1)

            avg = avg * 1e-9

            # long form:
            # diff_in_ns = x.diff().iloc[1:].astype('int64')
            # diff_in_seconds = diff_in_ns * 1e-9
            # avg = diff_in_seconds.mean()
            return avg
        return pd_avg_time_between


class Median(AggregationPrimitive):
    """Finds the median value of any feature with well-ordered values"""
    name = "median"
    input_types = [Numeric]
    return_type = None
    # max_stack_depth = 2

    def get_function(self):
        return np.median


class Skew(AggregationPrimitive):
    """Computes the skewness of a data set.
    For normally distributed data, the skewness should be about 0. A skewness
    value > 0 means that there is more weight in the left tail of the
    distribution.
    """
    name = "skew"
    input_types = [Numeric]
    return_type = Numeric
    stack_on = []
    stack_on_self = False
    # max_stack_depth = 1

    def get_function(self):
        return skew


class Std(AggregationPrimitive):
    """
    Finds the standard deviation of a numeric feature ignoring null values.
    """
    name = "std"
    input_types = [Numeric]
    return_type = Numeric
    # max_stack_depth = 2
    stack_on_self = False

    def get_function(self):
        return np.nanstd


class Last(AggregationPrimitive):
    """Returns the last value"""
    name = "last"
    input_types = [Variable]
    return_type = None
    stack_on_self = False
    # max_stack_depth = 1

    def get_function(self):
        def pd_last(x):
            return x.iloc[-1]
        return pd_last

'''
函数应用举例：
客户最近一年是否 买过**产品/登录过app/……
'''
class Any(AggregationPrimitive):
    """Test if any value is True"""
    name = "any"
    input_types = [Boolean]
    return_type = Boolean
    stack_on_self = False

    def get_function(self):
        return np.any

'''
函数应用举例：
客户最近n个月连续 买过**产品/登录过app/……
'''
class All(AggregationPrimitive):
    """Test if all values are True"""
    name = "all"
    input_types = [Boolean]
    return_type = Boolean
    stack_on_self = False

    def get_function(self):
        return np.all


'''
函数应用举例：
客户多久没有 买过**产品/登录过app/……
'''
class TimeSinceLast(AggregationPrimitive):
    """Time since last related instance"""
    name = "time_since_last"
    input_types = [DatetimeTimeIndex]
    return_type = Numeric
    uses_calc_time = True

    def get_function(self):

        def time_since_last(values, time=None):
            time_since = time - values.iloc[0]
            return time_since.total_seconds()

        return time_since_last


'''
函数应用举例：
客户资产下降/上升速度
'''
class Trend(AggregationPrimitive):
    """Calculates the slope of the linear trend of variable overtime"""
    name = "trend"
    input_types = [Numeric, DatetimeTimeIndex]
    return_type = Numeric

    def __init__(self, value, time_index, parent_entity, **kwargs):
        self.value = value
        self.time_index = time_index
        super(Trend, self).__init__([value, time_index],
                                    parent_entity,
                                    **kwargs)

    def get_function(self):
        def pd_trend(y, x):
            df = pd.DataFrame({"x": x, "y": y}).dropna()
            if df.shape[0] <= 2:
                return np.nan
            if isinstance(df['x'].iloc[0], (datetime, pd.Timestamp)):
                x = convert_datetime_to_floats(df['x'])
            else:
                x = df['x'].values

            if isinstance(df['y'].iloc[0], (datetime, pd.Timestamp)):
                y = convert_datetime_to_floats(df['y'])
            elif isinstance(df['y'].iloc[0], (timedelta, pd.Timedelta)):
                y = convert_timedelta_to_floats(df['y'])
            else:
                y = df['y'].values

            x = x - x.mean()
            y = y - y.mean()

            # prevent divide by zero error
            if len(np.unique(x)) == 1:
                return 0

            # consider scipy.stats.linregress for large n cases
            coefficients = np.polyfit(x, y, 1)

            return coefficients[0]
        return pd_trend


# # TODO: Not implemented yet
# class ConseqPos(AggregationPrimitive):
#     name = "conseq_pos"
#     input_types =  [(variable_types.Numeric,),
#                 (variable_types.Ordinal,)]
#     return_type = variable_types.Numeric
#     max_stack_depth = 1
#     stack_on = []
#     stack_on_exclude = []

#     def get_function(self):
#         raise NotImplementedError("This feature has not been implemented")


# # TODO: Not implemented yet
# class ConseqSame(AggregationPrimitive):
#     name = "conseq_same"
#     input_types =  [(variable_types.Categorical,),
#                 (variable_types.Ordinal,),
#                 (variable_types.Numeric,)]
#     return_type = variable_types.Numeric
#     max_stack_depth = 1
#     stack_on = []
#     stack_on_exclude = []

#     def get_function(self):
#         raise NotImplementedError("This feature has not been implemented")


# # TODO: Not implemented yet
# class TimeSinceLast(AggregationPrimitive):


def convert_datetime_to_floats(x):
    first = int(x.iloc[0].value * 1e-9)
    x = pd.to_numeric(x).astype(np.float64).values
    dividend = find_dividend_by_unit(first)
    x *= (1e-9 / dividend)
    return x


def convert_timedelta_to_floats(x):
    first = int(x.iloc[0].total_seconds())
    dividend = find_dividend_by_unit(first)
    x = pd.TimedeltaIndex(x).total_seconds().astype(np.float64) / dividend
    return x


def find_dividend_by_unit(time):
    """
    Finds whether time best corresponds to a value in
    days, hours, minutes, or seconds
    """
    for dividend in [86400., 3600., 60.]:
        div = time / dividend
        if round(div) == div:
            return dividend
    return 1
#%% md
transform primitive
#%%
rom .primitive_base import PrimitiveBase
from .utils import inspect_function_args
from featuretools.variable_types import (Discrete, Numeric, Boolean,
                                         Ordinal, Datetime, Timedelta,
                                         Variable, DatetimeTimeIndex, Id)
import datetime
import os
import pandas as pd
import numpy as np
import functools
current_path = os.path.dirname(os.path.realpath(__file__))
FEATURE_DATASETS = os.path.join(os.path.join(current_path, '..'),
                                'feature_datasets')


class TransformPrimitive(PrimitiveBase):
    """Feature for entity that is a based off one or more other features
        in that entity"""
    rolling_function = False

    def __init__(self, *base_features):
        # Any edits made to this method should also be made to the
        # new_class_init method in make_trans_primitive
        self.base_features = [self._check_feature(f) for f in base_features]
        if any(bf.expanding for bf in self.base_features):
            self.expanding = True
        assert len(set([f.entity for f in self.base_features])) == 1, \
            "More than one entity for base features"
        super(TransformPrimitive, self).__init__(self.base_features[0].entity,
                                                 self.base_features)

    def _get_name(self):
        name = u"{}(".format(self.name.upper())
        name += u", ".join(f.get_name() for f in self.base_features)
        name += u")"
        return name

    @property
    def default_value(self):
        return self.base_features[0].default_value


def make_trans_primitive(function, input_types, return_type, name=None,
                         description='A custom transform primitive',
                         cls_attributes=None, uses_calc_time=False,
                         associative=False):
    '''Returns a new transform primitive class
    Args:
        function (function): function that takes in an array  and applies some
            transformation to it.
        name (string): name of the function
        input_types (list[:class:`.Variable`]): variable types of the inputs
        return_type (:class:`.Variable`): variable type of return
        description (string): description of primitive
        cls_attributes (dict): custom attributes to be added to class
        uses_calc_time (bool): if True, the cutoff time the feature is being
            calculated at will be passed to the function as the keyword
            argument 'time'.
        associative (bool): If True, will only make one feature per unique set
            of base features
    Example:
        .. ipython :: python
            from featuretools.primitives import make_trans_primitive
            from featuretools.variable_types import Variable, Boolean
            def pd_is_in(array, list_of_outputs=None):
                if list_of_outputs is None:
                    list_of_outputs = []
                return pd.Series(array).isin(list_of_outputs)
            def isin_get_name(self):
                return u"%s.isin(%s)" % (self.base_features[0].get_name(),
                                         str(self.kwargs['list_of_outputs']))
            IsIn = make_trans_primitive(
                pd_is_in,
                [Variable],
                Boolean,
                name="is_in",
                description="For each value of the base feature, checks "
                "whether it is in a list that provided.",
                cls_attributes={"_get_name": isin_get_name})
    '''
    # dictionary that holds attributes for class
    cls = {"__doc__": description}
    if cls_attributes is not None:
        cls.update(cls_attributes)

    # creates the new class and set name and types
    name = name or function.func_name
    new_class = type(name, (TransformPrimitive,), cls)
    new_class.name = name
    new_class.input_types = input_types
    new_class.return_type = return_type
    new_class.associative = associative
    new_class, kwargs = inspect_function_args(new_class,
                                              function,
                                              uses_calc_time)

    if kwargs is not None:
        new_class.kwargs = kwargs

        def new_class_init(self, *args, **kwargs):
            self.base_features = [self._check_feature(f) for f in args]
            if any(bf.expanding for bf in self.base_features):
                self.expanding = True
            assert len(set([f.entity for f in self.base_features])) == 1, \
                "More than one entity for base features"
            self.kwargs.update(kwargs)
            self.partial = functools.partial(function, **self.kwargs)
            super(TransformPrimitive, self).__init__(
                self.base_features[0].entity, self.base_features)
        new_class.__init__ = new_class_init
        new_class.get_function = lambda self: self.partial
    else:
        # creates a lambda function that returns function every time
        new_class.get_function = lambda self, f=function: f

    return new_class


class IsNull(TransformPrimitive):
    """For each value of base feature, return true if value is null"""
    name = "is_null"
    input_types = [Variable]
    return_type = Boolean

    def get_function(self):
        return lambda array: pd.isnull(pd.Series(array))


class Absolute(TransformPrimitive):
    """Absolute value of base feature"""
    name = "absolute"
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return lambda array: np.absolute(array)


class TimeSincePrevious(TransformPrimitive):
    """Compute the time since the previous instance for each instance in a
     time indexed entity"""
    name = "time_since_previous"
    input_types = [DatetimeTimeIndex, Id]
    return_type = Numeric

    def __init__(self, time_index, group_feature):
        """Summary
        Args:
            base_feature (:class:`PrimitiveBase`): base feature
            group_feature (None, optional): variable or feature to group
                rows by before calculating diff
        """
        group_feature = self._check_feature(group_feature)
        assert issubclass(group_feature.variable_type, Discrete), \
            "group_feature must have a discrete variable_type"
        self.group_feature = group_feature
        super(TimeSincePrevious, self).__init__(time_index, group_feature)

    def _get_name(self):
        return u"time_since_previous_by_%s" % self.group_feature.get_name()

    def get_function(self):
        def pd_diff(base_array, group_array):
            bf_name = 'base_feature'
            groupby = 'groupby'
            grouped_df = pd.DataFrame.from_dict({bf_name: base_array,
                                                 groupby: group_array})
            grouped_df = grouped_df.groupby(groupby).diff()
            return grouped_df[bf_name].apply(lambda x:
                                             x.total_seconds())
        return pd_diff


class DatetimeUnitBasePrimitive(TransformPrimitive):
    """Transform Datetime feature into time or calendar units
     (second/day/week/etc)"""
    name = None
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        return lambda array: pd_time_unit(self.name)(pd.DatetimeIndex(array))


class TimedeltaUnitBasePrimitive(TransformPrimitive):
    """Transform Timedelta features into number of time units
     (seconds/days/etc) they encompass"""
    name = None
    input_types = [Timedelta]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd_time_unit(self.name)(pd.TimedeltaIndex(array))


class Day(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the day"""
    name = "day"


class Days(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of days"""
    name = "days"


class Hour(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the hour"""
    name = "hour"


class Hours(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of hours"""
    name = "hours"

    def get_function(self):
        def pd_hours(array):
            return pd_time_unit("seconds")(pd.TimedeltaIndex(array)) / 3600.
        return pd_hours


class Second(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the second"""
    name = "second"


class Seconds(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of seconds"""
    name = "seconds"


class Minute(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the minute"""
    name = "minute"


class Minutes(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of minutes"""
    name = "minutes"

    def get_function(self):
        def pd_minutes(array):
            return pd_time_unit("seconds")(pd.TimedeltaIndex(array)) / 60.
        return pd_minutes


class Week(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the week"""
    name = "week"


class Weeks(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of weeks"""
    name = "weeks"

    def get_function(self):
        def pd_weeks(array):
            return pd_time_unit("days")(pd.TimedeltaIndex(array)) / 7.
        return pd_weeks


class Month(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the month"""
    name = "month"


class Months(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of months"""
    name = "months"

    def get_function(self):
        def pd_months(array):
            return pd_time_unit("days")(pd.TimedeltaIndex(array)) * (12. / 365)
        return pd_months


class Year(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the year"""
    name = "year"


class Years(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of years"""
    name = "years"

    def get_function(self):
        def pd_years(array):
            return pd_time_unit("days")(pd.TimedeltaIndex(array)) / 365
        return pd_years


class Weekend(TransformPrimitive):
    """Transform Datetime feature into the boolean of Weekend"""
    name = "is_weekend"
    input_types = [Datetime]
    return_type = Boolean

    def get_function(self):
        return lambda df: pd_time_unit("weekday")(pd.DatetimeIndex(df)) > 4


class Weekday(DatetimeUnitBasePrimitive):
    """Transform Datetime feature into the boolean of Weekday"""
    name = "weekday"


# class Like(TransformPrimitive):
#     """Equivalent to SQL LIKE(%text%)
#        Returns true if text is contained with the string base_feature
#     """
#     name = "like"
#     input_types =  [(Text,), (Categorical,)]
#     return_type = Boolean

#     def __init__(self, base_feature, like_statement, case_sensitive=False):
#         self.like_statement = like_statement
#         self.case_sensitive = case_sensitive
#         super(Like, self).__init__(base_feature)

#     def get_function(self):
#         def pd_like(df, f):
#             return df[df.columns[0]].str.contains(f.like_statement,
#                                                   case=f.case_sensitive)
#         return pd_like


# class TimeSince(TransformPrimitive):
#     """
#     For each value of the base feature, compute the timedelta between it and
#     a datetime
#     """
#     name = "time_since"
#     input_types = [[DatetimeTimeIndex], [Datetime]]
#     return_type = Timedelta
#     uses_calc_time = True

#     def get_function(self):
#         def pd_time_since(array, time):
#             if time is None:
#                 time = datetime.now()
#             return (time - pd.DatetimeIndex(array)).values
#         return pd_time_since


def pd_time_since(array, time):
    if time is None:
        time = datetime.now()
    return (time - pd.DatetimeIndex(array)).values


TimeSince = make_trans_primitive(function=pd_time_since,
                                 input_types=[[DatetimeTimeIndex], [Datetime]],
                                 return_type=Timedelta,
                                 uses_calc_time=True,
                                 name="time_since")


class DaysSince(TransformPrimitive):
    """
    For each value of the base feature, compute the number of days between it
    and a datetime
    """
    name = "days_since"
    input_types = [DatetimeTimeIndex]
    return_type = Numeric
    uses_calc_time = True

    def get_function(self):
        def pd_days_since(array, time):
            if time is None:
                time = datetime.now()
            return pd_time_unit('days')(time - pd.DatetimeIndex(array))
        return pd_days_since



class IsIn(TransformPrimitive):
    """
    For each value of the base feature, checks whether it is in a list that is
    provided.
    """
    name = "isin"
    input_types = [Variable]
    return_type = Boolean

    def __init__(self, base_feature, list_of_outputs=None):
        self.list_of_outputs = list_of_outputs
        super(IsIn, self).__init__(base_feature)

    def get_function(self):
        def pd_is_in(array, list_of_outputs=self.list_of_outputs):
            if list_of_outputs is None:
                list_of_outputs = []
            return pd.Series(array).isin(list_of_outputs)
        return pd_is_in

    def _get_name(self):
        return u"%s.isin(%s)" % (self.base_features[0].get_name(),
                                 str(self.list_of_outputs))


class Diff(TransformPrimitive):
    """
    For each value of the base feature, compute the difference between it and
    the previous value.
    If it is a Datetime feature, compute the difference in seconds
    """
    name = "diff"
    input_types = [Numeric, Id]
    return_type = Numeric

    def __init__(self, base_feature, group_feature):
        """Summary
        Args:
            base_feature (:class:`PrimitiveBase`): base feature
            group_feature (:class:`PrimitiveBase`): variable or feature to
                group rows by before calculating diff
        """
        self.group_feature = self._check_feature(group_feature)
        super(Diff, self).__init__(base_feature, group_feature)

    def _get_name(self):
        base_features_str = self.base_features[0].get_name() + u" by " + \
            self.group_feature.get_name()
        return u"%s(%s)" % (self.name.upper(), base_features_str)

    def get_function(self):
        def pd_diff(base_array, group_array):
            bf_name = 'base_feature'
            groupby = 'groupby'
            grouped_df = pd.DataFrame.from_dict({bf_name: base_array,
                                                 groupby: group_array})
            grouped_df = grouped_df.groupby(groupby).diff()
            return grouped_df[bf_name]
        return pd_diff


class Not(TransformPrimitive):
    """
    For each value of the base feature, negates the boolean value.
    """
    name = "not"
    input_types = [Boolean]
    return_type = Boolean

    def _get_name(self):
        return u"NOT({})".format(self.base_features[0].get_name())

    def _get_op(self):
        return "__not__"

    def get_function(self):
        return lambda array: np.logical_not(array)


class Percentile(TransformPrimitive):
    """
    For each value of the base feature, determines the percentile in relation
    to the rest of the feature.
    """
    name = 'percentile'
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd.Series(array).rank(pct=True)


def pd_time_unit(time_unit):
    def inner(pd_index):
        return getattr(pd_index, time_unit).values
    return inner
#%%
