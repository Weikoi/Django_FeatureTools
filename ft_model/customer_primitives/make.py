#%%
import featuretools.primitives as primitives
#%%
# ?primitives
#%%

#%%

#%%

#%% md
# 自定义特征函数
#%%
import featuretools as ft
from featuretools.primitives import make_trans_primitive, Sum, Mean, Std
from featuretools.variable_types import Text, Numeric

#%%
def string_count(column, string=None):
    '''
    ..note:: this is a naive implementation used for clarity
    '''
    assert string is not None, "string to count needs to be defined"
    counts = [element.lower().count(string) for element in column]
    return counts

#%%

def string_count_get_name(self):
    return u"STRING_COUNT(%s, %s)" % (self.base_features[0].get_name(),
                                      '"'+str(self.kwargs['string']+'"'))


#%%
StringCount = make_trans_primitive(function=string_count,
                                   input_types=[Text],
                                   return_type=Numeric,
                                   cls_attributes={"get_name": string_count_get_name})

#%%
from featuretools.tests.testing_utils import make_ecommerce_entityset

es = make_ecommerce_entityset()
count_the_feat = StringCount(es['log']['comments'], string="the")

#%% md
# 原始日志数据
#%%
es['log'].df.head()
#%% md
# 统计日志表的评论字段出现the的求和值、平均值、标准差
#%%
feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="sessions",
                                      agg_primitives=[Sum, Mean, Std],
                                      seed_features=[count_the_feat])
feature_matrix[['STD(log.STRING_COUNT(comments, "the"))', 'SUM(log.STRING_COUNT(comments, "the"))', 'MEAN(log.STRING_COUNT(comments, "the"))']]
#%%
