import numpy as np
import pandas as pd
import featuretools as ft

data = ft.demo.load_mock_customer()

print(data['transactions']['amount'])
print(max(data['transactions']['amount']))

