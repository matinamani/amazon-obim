from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from analyzer import Analyzer
from statusTools import status


def make_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def rating_distribution(ratings, brand):
    path = './assets/plots/{}'.format(brand)
    make_directory(path)
    path = '{}/rating-distribution.png'.format(path)

    bars = tuple(ratings.keys())
    values = list(ratings.values())
    counter = np.arange(len(bars))

    plt.bar(counter, values)
    plt.xticks(counter, bars)

    plt.title('Rating Distribution ({})'.format(brand))
    plt.xlabel('Rating')
    plt.ylabel('Number of Reviews')

    plt.savefig(path, dpi=600)
    plt.clf()


def average_rating(brands):
    path = './assets/plots/overall/'
    make_directory(path)
    path = '{}/average-rating.png'.format(path)

    bars = tuple(brands.keys())
    values = list(brands.values())
    counter = np.arange(len(bars))

    plt.bar(counter, values)
    plt.xticks(counter, bars)

    plt.title('Average Rating')
    plt.xlabel('Brand Name')
    plt.ylabel('Average Rating')

    plt.savefig(path, dpi=600)
    plt.clf()


def sentiment_distribution(sentiments, brand, cleanup=False):
    path = './assets/plots/{}/cleaned'.format(
        brand) if cleanup else './assets/plots/{}'.format(brand)
    make_directory(path)
    path = '{}/sentiment-distribution.png'.format(path)
    bars = tuple(sentiments.keys())
    values = list(sentiments.values())
    counter = np.arange(len(bars))

    plt.bar(counter, values)
    plt.xticks(counter, bars)

    plt.title('Sentiments Distribution ({b}{clean})'.format(
        b=brand, clean=' cleaned' if cleanup else ''))
    plt.xlabel('Polarity')
    plt.ylabel('Number of Reviews')

    plt.savefig(path, dpi=600)
    plt.clf()


def average_sentiment(brands, cleanup=True):
    path = './assets/plots/overall/cleaned' if cleanup else './assets/plots/overall'
    make_directory(path)
    path = '{}/average-sentiment.png'.format(path)
    width = 0.25

    neg = []
    neu = []
    pos = []

    for brand in brands.keys():
        analysis = Analyzer(brand, stats=False, cleanup=cleanup)
        analysis.analyze()
        neg.append(analysis.avg_neg)
        neu.append(analysis.avg_neu)
        pos.append(analysis.avg_pos)

    r1 = np.arange(len(brands))
    r2 = [width + x for x in r1]
    r3 = [width + x for x in r2]

    plt.bar(r1, neg, width, color="#f73636", label='Negativity')
    plt.bar(r2, neu, width, color="#36d0f7", label='Neutrality')
    plt.bar(r3, pos, width, color="#29ff4c", label='Positivity')

    plt.title('Average Sentiments{}'.format(
        '(Cleaned Reviews)' if cleanup else ''))
    plt.xlabel('Brands')
    plt.ylabel('Percent')
    plt.xticks([width + r for r in range(len(brands))], brands)

    plt.legend()
    plt.savefig(path, dpi=600)
    plt.clf()


def comparison(brands, cleanup=True):
    path = './assets/plots/overall/cleaned' if cleanup else './assets/plots/overall'
    make_directory(path)
    path = '{}/comparison.png'.format(path)
    width = 0.25

    average_rating = []
    compound = []

    for brand in brands.keys():
        analysis = Analyzer(brand, stats=False, cleanup=cleanup)
        analysis.analyze()
        compound.append((analysis.avg_compound + 1) * 50)
        average_rating.append((analysis.average_rating - 1) * 25)

    r1 = np.arange(len(brands))
    r2 = [width + x for x in r1]

    plt.bar(r1, average_rating, width, color="#408ead", label='Average Rating')
    plt.bar(r2, compound, width, color="#49ba94", label='Sentiment Compound')

    plt.title('Comparison{}'.format('(Cleaned Reviews)' if cleanup else ''))
    plt.xlabel('Brands')
    plt.ylabel('Score out of 100')
    plt.xticks([(width / 2) + r for r in range(len(brands))], brands)

    plt.legend()
    plt.savefig(path, dpi=600)
    plt.clf()


def sentpol_ratepol(brand, cleanup=True):
    path = './assets/plots/{}/cleaned'.format(
        brand) if cleanup else './assets/plots/{}'.format(brand)
    make_directory(path)
    path = '{}/sentiment-polarity-vs-ratings-polarity.png'.format(path)
    width = 0.25
    ticks = ['Negative', 'Neutral', 'Positive']

    rating = [0, 0, 0]
    overall = [0, 0, 0]

    analysis = Analyzer(brand, stats=False, cleanup=cleanup)
    analysis.analyze()

    rating[0] = analysis.ratings['1'] + analysis.ratings['2']
    rating[1] = analysis.ratings['3']
    rating[2] = analysis.ratings['4'] + analysis.ratings['5']
    overall[0] = analysis.sentiments['negative']
    overall[1] = analysis.sentiments['neutral']
    overall[2] = analysis.sentiments['positive']

    r1 = np.arange(3)
    r2 = [width + x for x in r1]

    plt.bar(r1, rating, width, color="#408ead", label='# of Ratings')
    plt.bar(r2, overall, width, color="#49ba94", label='# of Sentiment')

    plt.title('Sentiment Polarity vs. Ratings Polarity{}'.format(
        '(Cleaned Reviews)' if cleanup else ''))
    plt.xlabel(brand)
    plt.ylabel('Count')
    plt.xticks([(width / 2) + r for r in range(3)], ticks)

    plt.legend()
    plt.savefig(path, dpi=600)
    plt.clf()
