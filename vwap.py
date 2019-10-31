
def vwap(close, high, low, volume, period):

    price = (close + high + low) / 3
    price_by_volume = price * volume

    sum_P_by_V = price_by_volume.rolling(period).sum()
    sum_V = volume.rolling(period).sum()

    vwap = sum_P_by_V / sum_V
    return vwap
