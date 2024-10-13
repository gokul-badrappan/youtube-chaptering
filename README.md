# Video Chaptering with Python

Video Chaptering is the process of dividing a video into distinct segments, each labeled with a specific title or chapter name, to enhance navigation and user experience. This project utilizes natural language processing (NLP) and machine learning techniques to automatically segment YouTube videos based on their content.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Setup](#setup)
- [Code Explanation](#code-explanation)
- [Usage](#usage)
- [Results](#results)
- [Author](#author)

## Introduction
In this project, we will extract audio from a YouTube video, transcribe it, and analyze the content to create chapters. Each chapter will contain the start time and a brief description based on the video content.

## Getting Started
To get started with Video Chaptering, you will need to collect data from a YouTube video using the YouTube Data API.

### Requirements
Make sure you have the following libraries installed:
- `google-api-python-client`
- `youtube-transcript-api`
- `pandas`
- `numpy`
- `matplotlib`
- `sklearn`

You can install them using pip:

```bash
pip install google-api-python-client youtube-transcript-api pandas numpy matplotlib scikit-learn
```

## Setup

	1.	Go to Google Cloud Console.
	2.	Create a new project.
	3.	Enable the YouTube Data API v3.
	4.	Create credentials and copy the generated API key.

## Code Explanation

The code is structured as follows:

	•	Import Libraries: The necessary libraries for data manipulation, visualization, and API access are imported at the beginning of the script.
	•	Function Definitions: Several functions are defined to handle different tasks:
	•	Extracting Video ID: A function that extracts the video ID from the provided YouTube URL.
	•	Fetching Video Title and Transcript: This function uses the YouTube Data API and the YouTube Transcript API to fetch the video’s title and its transcript.
	•	Saving Transcript: A function to save the fetched transcript data to a CSV file for later use.
	•	Data Processing: After retrieving the transcript, this part of the code processes the data by analyzing text lengths and identifying common words, which can help in summarizing the content.
	•	Topic Modeling: Using Non-negative Matrix Factorization (NMF), the script identifies key topics discussed in the video, helping to better organize the chapter segments.
	•	Chapter Creation: This section consolidates logical breaks in the transcript into broader chapters and generates meaningful chapter names based on the content.

## Usage

To use the script:

	1.	Run the script.
	2.	Enter the YouTube video link when prompted.

## Results

The script saves the transcript and chapter information in a CSV file, with each chapter labeled with its start time and title.


## Author
Gokul B
