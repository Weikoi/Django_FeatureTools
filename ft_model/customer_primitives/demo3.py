import pandas as pd


def rise_count(vals, n=2):
    count = 0
    length = len(vals)
    start = 0
    end = 1

    new_s = vals.shift(-1) - vals
    print(new_s)
    for _ in range(n):
        start_flag = length // n * start
        end_flag = length // n * end
        print(start_flag, end_flag)
        piece = new_s.iloc[start_flag:end_flag]
        if (sum(piece) > 0):
            count += 1
        start += 1
        end += 1
    return count


if __name__ == '__main__':
    s = pd.Series([0, 2, 3, 10, 20, 5])
    print(rise_count(s))
