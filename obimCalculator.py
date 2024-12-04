import pandas as pd
from pandas.core.algorithms import unique
from normalization import Normalizer
from vader import Vader
from reviewCleaner import cleaner
from statusTools import status


def OBIM(brand, aspects=['phone', 'camera'], stats=True, cleanup=True):
    if stats:
        status(f'Calculating OBIM for brand {brand}...')
    vader = Vader()
    aspects = {aspect: 0 for aspect in aspects}
    all_reviews = pd.read_csv(
        f'./assets/reviews/{brand}.csv')['Review'].map(cleaner) if cleanup else pd.read_csv(
        f'./assets/reviews/{brand}.csv')['Review'].map(lambda r: r.strip().lower())

    normalizer = Normalizer(brand, aspects, stats=False, cleanup=cleanup)
    normalizer.normalize()

    for aspect in aspects.keys():
        reviews = all_reviews[all_reviews.apply(lambda r: aspect in r)]
        compound = reviews.apply(
            lambda r: (vader.sentiment_analysis(r)[-1] + 1) / 2).mean()
        strength = normalizer.strength[aspect]
        uniqueness = normalizer.uniqueness[aspect]
        aspects[aspect] = compound * strength * uniqueness

    obim = sum(aspects.values())
    return obim, aspects
