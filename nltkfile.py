import nltk
nltk.download('punkt')  # Example corpus, replace 'punkt' with the corpora you need

# List of corpora you want to download
corpora_to_download = [
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    # Add more corpora as needed
]

# Download each corpus
for corpus in corpora_to_download:
    nltk.download(corpus)