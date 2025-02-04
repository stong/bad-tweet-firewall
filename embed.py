import json
from openai import OpenAI
from tqdm import tqdm  # For progress bar
import time
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Load tweets
with open('tweets_export.json', 'r', encoding='utf-8') as f:
    tweets = json.load(f)

import os
if not os.path.isfile('tweet_embeddings.json'):
    embeddings_dict = {}
else:
    with open('tweet_embeddings.json', 'r') as f:
        embeddings_dict = json.load(f)

print('embeddings count:',len(embeddings_dict))
tweets = list(filter(lambda tweet: len(tweet['text']) > 0, tweets))
tweets = list(filter(lambda tweet: tweet['tweetId'] not in embeddings_dict, tweets))

BATCH_SIZE = 32

# Process tweets in batches
for i in tqdm(range(0, len(tweets), BATCH_SIZE)):
    batch = tweets[i:i + BATCH_SIZE]
    
    try:
        inp = [tweet['username'] + ": " + tweet['text'] for tweet in batch]
        # Get embeddings for batch of tweets
        response = client.embeddings.create(
            input=inp,
            model="text-embedding-3-small",
            dimensions=256
        )
        
        # Store embeddings in dictionary
        for tweet, embedding_data in zip(batch, response.data):
            embeddings_dict[tweet['tweetId']] = embedding_data.embedding
        
        # Small sleep between batches
        time.sleep(0.1)
        
    except Exception as e:
        print()
        print(f"Error processing batch starting at index {i}: {str(e)}")
        print(inp)
        continue

# Save embeddings to file
with open('tweet_embeddings.json', 'w', encoding='utf-8') as f:
    json.dump(embeddings_dict, f)

print('embeddings count:',len(embeddings_dict))
print(f"Processed {len(embeddings_dict)} tweets successfully")
