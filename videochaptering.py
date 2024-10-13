# Import necessary libraries
import re
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

# Replace with your YouTube Data API key
API_KEY = 'Your API Key'

def get_video_id(url):
    """
    Extract the video ID from the provided YouTube URL.

    Args:
    url (str): The YouTube video URL.

    Returns:
    str: The extracted video ID or None if invalid.
    """
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id_match.group(1) if video_id_match else None

def get_video_title(video_id):
    """
    Fetch the title of the YouTube video using its ID.

    Args:
    video_id (str): The YouTube video ID.

    Returns:
    str: The title of the video or 'Unknown Title' if not found.
    """
    # Build the YouTube service
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    # Fetch video details
    request = youtube.videos().list(part='snippet', id=video_id)
    response = request.execute()

    # Extract and return the title
    title = response['items'][0]['snippet']['title'] if response['items'] else 'Unknown Title'
    return title

def get_video_transcript(video_id):
    """
    Retrieve the transcript of the YouTube video.

    Args:
    video_id (str): The YouTube video ID.

    Returns:
    list: A list of transcript entries or an empty list if an error occurs.
    """
    # Fetch the transcript
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def save_to_csv(title, transcript, filename):
    """
    Save the video title and transcript to a CSV file.

    Args:
    title (str): The title of the video.
    transcript (list): The transcript data.
    filename (str): The name of the CSV file to save.
    """
    # Prepare the transcript data for saving
    transcript_data = [{'start': entry['start'], 'text': entry['text']} for entry in transcript]
    df = pd.DataFrame(transcript_data)
    df.to_csv(filename, index=False)

    # Save the title separately
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Title:', title])

def main():
    """
    Main function to execute the video chaptering process.
    """
    # Input video URL from the user
    url = input('Enter the YouTube video link: ')
    video_id = get_video_id(url)

    # Validate video ID
    if not video_id:
        print('Invalid YouTube URL.')
        return

    # Fetch video title and transcript
    title = get_video_title(video_id)
    transcript = get_video_transcript(video_id)

    # Check if transcript is available
    if not transcript:
        print('No transcript available for this video.')
        return

    # Define filename for saving the transcript
    filename = f"{video_id}_transcript.csv"
    save_to_csv(title, transcript, filename)
    print(f'Transcript saved to {filename}')

    # Load the saved transcript data
    transcript_df = pd.read_csv(filename)
    transcript_df['start'] = pd.to_numeric(transcript_df['start'], errors='coerce')

    # Display basic statistics about the transcript
    print("Dataset Overview:")
    print(transcript_df.info())
    print("\nBasic Statistics:")
    print(transcript_df.describe())

    # Analyze the distribution of text lengths
    transcript_df['text_length'] = transcript_df['text'].apply(len)
    plt.figure(figsize=(10, 5))
    plt.hist(transcript_df['text_length'], bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.show()

    # Find and plot the most common words
    vectorizer = CountVectorizer(stop_words='english')
    word_counts = vectorizer.fit_transform(transcript_df['text'])
    word_counts_df = pd.DataFrame(word_counts.toarray(), columns=vectorizer.get_feature_names_out())
    common_words = word_counts_df.sum().sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 5))
    common_words.plot(kind='bar', color='green', alpha=0.7)
    plt.title('Top 20 Common Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.show()

    # Topic modeling using NMF
    n_topics = 10
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(transcript_df['text'])
    nmf = NMF(n_components=n_topics, random_state=42).fit(tf)

    def display_topics(model, feature_names, no_top_words):
        """
        Display the top words for each topic.

        Args:
        model: The NMF model.
        feature_names (list): The feature names.
        no_top_words (int): Number of top words to display.
        """
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
            topics.append(" ".join(topic_words))
        return topics

    topics = display_topics(nmf, tf_vectorizer.get_feature_names_out(), 10)
    print("\nIdentified Topics:")
    for i, topic in enumerate(topics):
        print(f"Topic {i + 1}: {topic}")

    # Get topic distribution for each text segment
    topic_distribution = nmf.transform(tf)
    topic_distribution_trimmed = topic_distribution[:len(transcript_df)]
    transcript_df['dominant_topic'] = topic_distribution_trimmed.argmax(axis=1)

    # Identify logical breaks based on topic changes
    logical_breaks = []
    for i in range(1, len(transcript_df)):
        if transcript_df['dominant_topic'].iloc[i] != transcript_df['dominant_topic'].iloc[i - 1]:
            logical_breaks.append(transcript_df['start'].iloc[i])

    # Consolidate logical breaks into broader chapters
    threshold = 60  # seconds
    consolidated_breaks = []
    last_break = None

    for break_point in logical_breaks:
        if last_break is None or break_point - last_break >= threshold:
            consolidated_breaks.append(break_point)
            last_break = break_point

    # Merge consecutive breaks with the same dominant topic
    final_chapters = []
    last_chapter = (consolidated_breaks[0], transcript_df['dominant_topic'][0])

    for break_point in consolidated_breaks[1:]:
        current_topic = transcript_df[transcript_df['start'] == break_point]['dominant_topic'].values[0]
        if current_topic == last_chapter[1]:
            last_chapter = (last_chapter[0], current_topic)
        else:
            final_chapters.append(last_chapter)
            last_chapter = (break_point, current_topic)

    final_chapters.append(last_chapter)  # Append the last chapter

    # Convert the final chapters to a readable time format and generate names
    chapter_points = []
    chapter_names = []

    for i, (break_point, topic_idx) in enumerate(final_chapters):
        chapter_time = pd.to_datetime(break_point, unit='s').strftime('%H:%M:%S')
        chapter_points.append(chapter_time)

        # Get the context for the chapter name
        chapter_text = transcript_df[(transcript_df['start'] >= break_point) & (transcript_df['dominant_topic'] == topic_idx)]['text'].str.cat(sep=' ')

        # Extract key phrases to create a chapter name
        vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
        tfidf_matrix = vectorizer.fit_transform([chapter_text])
        feature_names = vectorizer.get_feature_names_out()
        chapter_name = " ".join(feature_names)

        chapter_names.append(f"Chapter {i+1}: {chapter_name}")

    # Display the final chapter points with names
    print("\nFinal Chapter Points with Names:")
    for time, name in zip(chapter_points, chapter_names):
        print(f"{time} - {name}")

if __name__ == '__main__':
    main()
