from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class Vader:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def sentiment_analysis(self, sentence):
        result = self.sentiment_analyzer.polarity_scores(sentence)

        positive = result['pos'] * 100
        neutral = result['neu'] * 100
        negative = result['neg'] * 100
        compound = result['compound']

        overall = 'positive' if compound >= 0.05 else (
            'negative' if compound <= -0.05 else 'neutral')

        return positive, neutral, negative, overall, compound
