# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def load_data(file_path):
    """Load and preprocess the dataset"""
    df = pd.read_csv(file_path)
    return df.rename(columns={'Text': 'text', 'Sentiment': 'sentiment'})

def display_sentiment_counts(df):
    """Display sentiment value counts"""
    print("\nSentiment Counts:")
    print(df['sentiment'].value_counts())

def plot_sentiment_distribution(df):
    """Create sentiment distribution plot"""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='sentiment', hue='sentiment', palette='pastel', legend=False)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def create_wordcloud(df):
    """Create wordcloud for positive sentiment"""
    positive_texts = df[df['sentiment'] == 'Positive']['text'].dropna()
    positive_text = " ".join(positive_texts)

    if positive_text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Positive Sentiment Posts')
        plt.tight_layout()
        plt.show()
    else:
        print("No positive text found. Skipping word cloud.")

def export_data(df, output_path):
    """Export cleaned data to CSV"""
    columns = ['text', 'sentiment', 'Timestamp', 'Platform', 'Hashtags', 'Country']
    df[columns].to_csv(output_path, index=False)
    print("Final dataset exported successfully!")

def main():
    # File paths
    input_path = r'c:\Users\Prince\Downloads\internship\repos\FUTURE_DS_01\sentimentdataset.csv'
    output_path = r'c:\Users\Prince\Downloads\internship\repos\FUTURE_DS_01\final_sentiment_data.csv'

    # Load and process data
    df = load_data(input_path)
    
    # Generate visualizations
    display_sentiment_counts(df)
    plot_sentiment_distribution(df)
    create_wordcloud(df)
    
    # Export processed data
    export_data(df, output_path)

if __name__ == "__main__":
    main()