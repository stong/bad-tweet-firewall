# Bad Tweet Firewall

Highlights bad tweets in a big red border. You train your personal algorithm to decide what tweets get highlighted.

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
