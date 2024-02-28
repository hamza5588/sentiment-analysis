
    'maxent_ne_chunker',
    # Add more corpora as needed
]

# Download each corpus
for corpus in corpora_to_download:
    nltk.download(corpus)
