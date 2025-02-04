// Database setup and initialization
const DB_NAME = 'TweetDB';
const DB_VERSION = 1;
let db;

const processedTweets = new Set();
let currentlyHoveredTweet = null; // Keep track of which tweet is being hovered

class LRUCache {
    constructor(capacity = 100) {
        this.capacity = capacity;
        this.cache = new Map();
    }

    // Get value by key
    get(key) {
        if (!this.cache.has(key)) {
            return undefined;
        }
        
        // Move accessed item to the end (most recently used)
        const value = this.cache.get(key);
        this.cache.delete(key);
        this.cache.set(key, value);
        return value;
    }

    // Set value by key
    set(key, value) {
        // If key exists, update its value and move it to the end
        if (this.cache.has(key)) {
            this.cache.delete(key);
        }
        // If cache is at capacity, remove the least recently used item
        else if (this.cache.size >= this.capacity) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        
        this.cache.set(key, value);
    }

    // Get current size of cache
    size() {
        return this.cache.size;
    }

    // Clear the cache
    clear() {
        this.cache.clear();
    }

    // Check if key exists in cache
    has(key) {
        return this.cache.has(key);
    }

    // Get all keys in order (least recently used first)
    keys() {
        return Array.from(this.cache.keys());
    }

    // Get all values in order (least recently used first)
    values() {
        return Array.from(this.cache.values());
    }
}
const tweetScoresCache = new LRUCache(100);

function initDB() {
    return new Promise((resolve, reject) => {
        const request = window.indexedDB.open(DB_NAME, DB_VERSION);
        
        request.onerror = () => reject(request.error);
        
        request.onsuccess = (event) => {
            db = event.target.result;
            resolve(db);
        };
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            
            // Create tweets store (table)
            if (!db.objectStoreNames.contains('tweets')) {
                const tweetStore = db.createObjectStore('tweets', { keyPath: 'tweetId' });
                tweetStore.createIndex('timestamp', 'timestamp', { unique: false });
            }
            
            // Create ratings store (table)
            if (!db.objectStoreNames.contains('ratings')) {
                db.createObjectStore('ratings', { keyPath: 'tweetId' });
            }
            // Create ratings store (table)
            if (!db.objectStoreNames.contains('harmful')) {
                db.createObjectStore('harmful', { keyPath: 'tweetId' });
            }
        };
    });
}

function getTweetUserDisplayName(article) {
    const userNameElement = article.querySelector('[data-testid="User-Name"]');
    const displayName = userNameElement ? userNameElement.textContent.split('@')[0].trim() : '';
    let username = userNameElement ? '@' + userNameElement.textContent.split('@')[1]?.trim() : '';
    username = username.includes('·') ? username.split('·')[0] : username; // strip timestamp
    return {
        username: username,
        displayName: displayName
    }
}

async function saveTweet(article) {
    const tweetId = getTweetId(article);
    if (!tweetId) return;

    // Get user information
    userDisplayName = getTweetUserDisplayName(article);

    const tweetData = {
        tweetId,
        text: getTweetText(article),
        displayName: userDisplayName.displayName,
        username: userDisplayName.username,
        timestamp: new Date().toISOString(),
        page: window.location.href
    };
    console.log(tweetData);

    const transaction = db.transaction(['tweets'], 'readwrite');
    const store = transaction.objectStore('tweets');
    
    return new Promise((resolve, reject) => {
        const request = store.put(tweetData);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
    });
}

async function saveRating(tweetId, rating) {
    const transaction = db.transaction(['ratings'], 'readwrite');
    const store = transaction.objectStore('ratings');
    
    return new Promise((resolve, reject) => {
        const request = store.put({ tweetId, rating });
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
    });
}

async function saveHarmful(tweetId, rating) {
    const transaction = db.transaction(['harmful'], 'readwrite');
    const store = transaction.objectStore('harmful');
    
    return new Promise((resolve, reject) => {
        const request = store.put({ tweetId, rating });
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
    });
}

function getTweetId(article) {
    const timeElement = article.querySelector('time');
    if (timeElement) {
        const linkElement = timeElement.closest('a');
        if (linkElement) {
            return linkElement.href;
        }
    }
    
    const tweetLink = article.querySelector('a[href*="/status/"]');
    if (tweetLink) {
        return tweetLink.href;
    }
    
    return null;
}

function getTweetText(article) {
    const tweetText = Array.from(article.querySelectorAll('[data-testid="tweetText"]'))
        .map(element => element.textContent)
        .join(' ');
    return tweetText;
}

// Add mouse enter/leave listeners to tweets
function addHoverListeners(tweet) {
    tweet.addEventListener('mouseenter', () => {
        currentlyHoveredTweet = tweet;
    });
    
    tweet.addEventListener('mouseleave', () => {
        if (currentlyHoveredTweet === tweet) {
            currentlyHoveredTweet = null;
        }
    });
}

async function scanTweetForHarmful(tweet) {
    const tweetId = getTweetId(tweet);
    if (!tweetScoresCache.has(tweetId)) {
        const score = await getTweetHarmScore(tweet);
        tweetScoresCache.set(tweetId, score);
        console.log(`Tweet ${tweetId} is has score ${score}`);
    }
    const score = tweetScoresCache.get(tweetId);
    if (score < 0.01) {
        tweet.style.borderWidth = '5px';
        tweet.style.borderColor = 'red';
        // await saveHarmful(tweetId, score);
    }
}

// Modified processTweets to save tweets to DB
async function processTweets() {
    const tweets = document.querySelectorAll('article');
    for (const tweet of tweets) {
        const tweetId = getTweetId(tweet);
        if (!tweet._hasHoverListener) {
            tweet._hasHoverListener = true;
            addHoverListeners(tweet);
            await scanTweetForHarmful(tweet);
        }
        if (tweetId && !processedTweets.has(tweetId)) {
            processedTweets.add(tweetId);
            try {
                await saveTweet(tweet);
            } catch (error) {
                console.error('Error saving tweet:', error);
            }
        }
    }
}

// openai shit

const tweet = { username: "user1", text: "some tweet" };
const input = `${tweet.username}: ${tweet.text}`;

async function getTextEmbedding(tweetText) {
    try {
        const response = await fetch('https://api.openai.com/v1/embeddings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + OPENAI_API_KEY
            },
            body: JSON.stringify({
                input: tweetText,  // Fixed: was using 'input' instead of tweetText
                model: "text-embedding-3-small",
                dimensions: 256
            })
        });
        
        const data = await response.json();
        return data.data[0].embedding;  // OpenAI returns embeddings in data[0].embedding
    } catch (error) {
        console.error('Error:', error);
        throw error;  // Re-throw to handle it in the calling function
    }
}

// classifier shit

async function loadONNXModel() {
    try {
        const session = await ort.InferenceSession.create(chrome.runtime.getURL('tweet_regressor.onnx'));
        return session;
    } catch (e) {
        console.error("Failed to load ONNX model:", e);
    }
}

async function predict(session, embedding) {
    try {
        // Convert embedding to float32 tensor
        const tensor = new ort.Tensor('float32', new Float32Array(embedding), [1, 256]);
        
        // Run inference
        const feeds = { input: tensor };
        const results = await session.run(feeds);
        
        // Get prediction (results.output is a tensor)
        const prediction = results.output.data[0];
        return prediction;
    } catch (e) {
        console.error("Prediction failed:", e);
        return null;
    }
}


// Usage example:
let model = null;

// Load model when extension starts
loadONNXModel().then(loadedModel => {
    model = loadedModel;
    console.log("Model loaded successfully!");
});

// Example function to classify a tweet
async function classifyEmbedding(embedding) {
    if (!model) {
        console.error("Model not loaded yet!");
        return;
    }
    
    const prediction = await predict(model, embedding);
    return prediction; // Will be between 0 and 1
}

async function getTweetHarmScore(tweet) {
    const username = getTweetUserDisplayName(tweet).username;
    const tweetText = getTweetText(tweet);
    const embedding = await getTextEmbedding(username + ": " + tweetText);
    const classifierOutput = await classifyEmbedding(embedding);
    console.log(classifierOutput);
    return classifierOutput;
}

// Modified keyboard event listener to save ratings
document.addEventListener('keypress', async (event) => {
    if ((event.key === '1' || event.key === '2' || event.key == '3' || event.key == '0') && currentlyHoveredTweet) {
        const tweetId = getTweetId(currentlyHoveredTweet);
        if (tweetId) {
            if (event.key == '3') {
                const harmScore = await getTweetHarmScore(currentlyHoveredTweet);
                alert(harmScore);
            } else { // 1 or 2 or 0
                const rating = parseInt(event.key);
                try {
                    await saveRating(tweetId, rating);
                    alert(`${tweetId} ${event.key}`);
                    console.log(`Saved rating ${rating} for tweet ${tweetId}`);
                } catch (error) {
                    console.error('Error saving rating:', error);
                }
            }
        }
    }
});

// Initialize DB before starting the observer
initDB().then(() => {
    if (window.MutationObserver) {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach(mutation => {
                if (mutation.addedNodes.length) {
                    processTweets();
                }
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
    
    processTweets();
}).catch(error => {
    console.error('Failed to initialize database:', error);
});