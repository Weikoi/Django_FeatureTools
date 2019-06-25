
import featuretools as ft
import pickle as pk

data = ft.demo.load_mock_customer()

data_demo = {"transactions": data["transactions"], "sessions": data["sessions"]}

pk.dump(data_demo, file=open("data_transactions_sessions.pkl", "wb"))
