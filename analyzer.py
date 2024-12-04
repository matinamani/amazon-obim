import pandas as pd
from vader import Vader
from reviewCleaner import cleaner
from statusTools import *


class Analyzer:
    def __init__(self, brand, stats=True, cleanup=True):
        self.stats = stats
        self.cleanup = cleanup
        self.vader = Vader()
        self.brand = brand
        self.df = pd.read_csv(f'./assets/reviews/{brand}.csv')
        self.reviews = self.df['Review'].map(lambda r: r.strip().lower())
        self.reviews = self.reviews.apply(cleaner) if cleanup else self.reviews

    def analyze(self):
        if self.stats:
            status(f'Analyzing {self.brand} reviews...')

        self.calc_total_average()
        self.calc_rating_counts()
        self.anal_sentiments()

    def calc_total_average(self):
        self.average_rating = self.df['Rating'].mean()

    def calc_rating_counts(self):
        self.ratings = self.df.groupby('Rating').size().to_dict()

    def anal_sentiments(self):
        sent = self.reviews.apply(
            lambda r: self.vader.sentiment_analysis(r)[3])
        self.sentiments = sent.groupby(sent).size().to_dict()
        self.avg_pos = self.reviews.apply(
            lambda r: self.vader.sentiment_analysis(r)[0]).mean()
        self.avg_neu = self.reviews.apply(
            lambda r: self.vader.sentiment_analysis(r)[1]).mean()
        self.avg_neg = self.reviews.apply(
            lambda r: self.vader.sentiment_analysis(r)[2]).mean()
        self.avg_compound = self.reviews.apply(
            lambda r: self.vader.sentiment_analysis(r)[4]).mean()

    def show_info(self):
        print(f'Analysis results for {self.brand}')
        print(
            f'(results from {len(self.reviews)} reviews, cleanup = {self.cleanup})\n')
        print('--Rating Results--')
        print(f'Avg Rating: {self.average_rating}')
        print('Rating Counts:', self.ratings, '\n')
        print('--Sentiment Results--')
        print(f'Avg pos: {self.avg_pos}')
        print(f'Avg neu: {self.avg_neu}')
        print(f'Avg neg: {self.avg_neg}')
        print(f'Avg Compound: {self.avg_compound}')
        print('Overall sentiments: ', self.sentiments, '\n')
        print('-----------------------------------------------------\n')
