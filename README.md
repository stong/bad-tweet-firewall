# Bad Tweet Firewall

Highlights bad tweets in a big red border. You train your personal algorithm to decide what tweets get highlighted.

The idea is that negative emotions from your feed generally harm your mental health. Meanwhile, most tweets are just honestly not that valuable. So let's make a firewall to just filter out posts with bad attitude and vibes.

## Examples

These are all examples from my personal curated algorithm. You can train the extension to filter whatever you want. Note that I do not have any beef with the particular posters I just don't like seeing negative things on my timeline because they make me sad.

**Political content:**

![image](https://github.com/user-attachments/assets/9de61b2b-c72e-498c-99d3-9009f635e26d)

**Unsavory celebrity gossip with negative sentiment:**

![image](https://github.com/user-attachments/assets/c1ae48d7-d002-4dc8-b880-afdcf173aba2)

**Negative sentiment hot takes:**

![image](https://github.com/user-attachments/assets/0315bfe4-e45d-4736-b558-b128b5208708)

**Commercial announcements:**

![image](https://github.com/user-attachments/assets/380533f0-7134-4464-877d-0bb0642a90cb)

![image](https://github.com/user-attachments/assets/c978988f-352f-4dba-a6dc-16248544a504)

**AGI doomposting**

![image](https://github.com/user-attachments/assets/5fae843c-5f42-4bf3-90dc-979fb147ad2b)

**Engagement bait**

![image](https://github.com/user-attachments/assets/a2ff387d-788d-4d52-ae7c-caedbca70524)

**general Doomposting**

![image](https://github.com/user-attachments/assets/bf8fecb6-194a-4086-ae0c-c7c6f05a3e73)

## How does it work?

Tweet username + content -> OpenAI `text-embedding-3-small` -> 256 dimensions embedding -> binary classifier

# How to personalize (train) your own algorithm instructions

Extension lets you label tweets as bad or neutral. Then you save this data and rerun the training script.

0. populate `Source/api-key.js` with: `const OPENAI_API_KEY = 'sk-proj-...';`
1. Install the extension, go on the timeline
2. To mark a tweet as bad, hover the tweet and press "1" on keyboard
3. To mark a tweet as always safe, hover the tweet and press "2" on keyboard
3. To mark a tweet as safe, hover the tweet and press "0" on the keyboard
3. You can also press "3" to inspect the score of the hovered tweet
4. Unmarked tweets are default to safe
5. Extract training data: copy paste contents of helper.js into js console on x.com
6. Copy downloaded files into this directory
6. populate `.env` with `OPENAI_API_KEY=...`
7. Run embed.py to (re-)compute embeddings with openai. If you add more training data later, it will not recompute embedding you already have
8. Run train.py to produce a pytorch model
9. Run export-model-to-onnx.py to convert pytorch model to ONNX and place it in browser extension folder (`Source/`)
10. Reload the browser extension, now you have loaded the newly trained model
