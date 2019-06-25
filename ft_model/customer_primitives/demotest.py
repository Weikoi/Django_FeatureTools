import featuretools

transactions_types = ['featuretools.variable_types.Id', 'featuretools.variable_types.Id',
                      'featuretools.variable_types.TimeIndex', 'featuretools.variable_types.Id',
                      'featuretools.variable_types.Numeric']

sessions_types = ['featuretools.variable_types.Id', 'featuretools.variable_types.Id',
                  'featuretools.variable_types.Categorical', 'featuretools.variable_types.Datetime']

customers_types = ['featuretools.variable_types.Id', 'featuretools.variable_types.Categorical',
                   'featuretools.variable_types.Datetime', 'featuretools.variable_types.Datetime']

products_types = ['featuretools.variable_types.Id', 'featuretools.variable_types.Categorical']

transactions_columns = ['transaction_id', 'session_id', 'transaction_time', 'product_id', 'amount']

sessions_columns = ['session_id', 'customer_id', 'device', 'session_start']

customers_columns = ['customer_id', 'zip_code', 'join_date', 'date_of_birth']

products_columns = ['product_id', 'brand']

print(transactions_columns)
print(sessions_columns)
print(customers_columns)
print(products_columns)

type_dict1 = {k: v for k, v in zip(transactions_columns, transactions_types)}
print(type_dict1)
# type_dict2 = {k: eval(v) for k in sessions_columns for v in sessions_types}
# type_dict3 = {k: eval(v) for k in customers_columns for v in customers_types}
# type_dict4 = {k: eval(v) for k in products_columns for v in products_types}
#
# type_dict = {}
# type_dict.update(type_dict1)
# type_dict.update(type_dict2)
# type_dict.update(type_dict3)
# type_dict.update(type_dict4)
# print("type_dict", type_dict)
