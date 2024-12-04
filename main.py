from os import system

from analyzer import Analyzer
from normalization import Normalizer
from obimCalculator import OBIM
from plotter import *
from statusTools import *

brands = {
    "apple": 0,
    "google": 0,
    "huawei": 0,
    "motorola": 0,
    "nokia": 0,
    "samsung": 0,
    "sony": 0,
    "xiaomi": 0,
}

aspects = [
    "phone",
    "screen",
    "battery",
    "camera",
    "charger",
    "charge",
    "service",
    "product",
    "device",
    "experience",
    "price",
    "sound",
]

for brand in brands.keys():
    analyzer = Analyzer(brand)
    analyzer.analyze()
    analyzer.show_info()
