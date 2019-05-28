import featuretools as ft
import featuretools
data = ft.demo.load_mock_customer()

transactions_df = data["transactions"].merge(data["sessions"]).merge(data["customers"]).merge(data["products"])

es = ft.EntitySet(id="customer_data")
es = es.entity_from_dataframe(entity_id="transactions",
                              dataframe=transactions_df,
                              index="transaction_id",
                              time_index="transaction_time",
                              variable_types={"product_id": ft.variable_types.Categorical,
                                              "zip_code": ft.variable_types.ZIPCode})
print(es["transactions"].variables)
sessions_columns = list(data["sessions"].columns)
print(sessions_columns)
sessions_columns.remove("session_id")
sessions_columns.remove("session_start")

es = es.normalize_entity(base_entity_id="transactions",
                         new_entity_id="sessions",
                         index="session_id",
                         make_time_index="session_start",
                         additional_variables=["device", "customer_id", "zip_code", "session_start", "join_date"])
print(type(featuretools.variable_types.Numeric))
print(type(eval("featuretools.variable_types."+"Numeric")))
