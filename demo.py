import re
a = [1.0, 2.0, 1.03, 2.03, 3.004]
return_list = []

for i in a:
    if i is None:
        return_list.append(None)
    elif re.search(r"\.0\b", str(i)):
        return_list.append(int(i))
    else:
        return_list.append(round(i, 2))

print(return_list)