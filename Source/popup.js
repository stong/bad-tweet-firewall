const DB_NAME = 'TweetDB';
const DB_VERSION = 1;

// Function to open database and get data in one go
async function getDataFromDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onerror = () => reject(request.error);
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            
            // Create stores if they don't exist
            if (!db.objectStoreNames.contains('tweets')) {
                const tweetStore = db.createObjectStore('tweets', { keyPath: 'tweetId' });
                tweetStore.createIndex('timestamp', 'timestamp', { unique: false });
            }
            
            if (!db.objectStoreNames.contains('ratings')) {
                db.createObjectStore('ratings', { keyPath: 'tweetId' });
            }
        };
        
        request.onsuccess = async (event) => {
            const db = event.target.result;
            try {
                const data = {
                    tweets: await getAllFromStore(db, 'tweets'),
                    ratings: await getAllFromStore(db, 'ratings')
                };
                console.log(data);
                resolve(data);
            } catch (error) {
                reject(error);
            } finally {
                db.close();
            }
        };
    });
}

function getAllFromStore(db, storeName) {
    return new Promise((resolve, reject) => {
        const transaction = db.transaction([storeName], 'readonly');
        const store = transaction.objectStore(storeName);
        const request = store.getAll();

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function clearDatabase() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        
        request.onsuccess = async (event) => {
            const db = event.target.result;
            try {
                const transaction1 = db.transaction(['tweets'], 'readwrite');
                const transaction2 = db.transaction(['ratings'], 'readwrite');
                
                await Promise.all([
                    clearStore(transaction1, 'tweets'),
                    clearStore(transaction2, 'ratings')
                ]);
                
                resolve();
            } catch (error) {
                reject(error);
            } finally {
                db.close();
            }
        };
        
        request.onerror = () => reject(request.error);
    });
}

function clearStore(transaction, storeName) {
    return new Promise((resolve, reject) => {
        const store = transaction.objectStore(storeName);
        const request = store.clear();
        
        transaction.oncomplete = () => resolve();
        transaction.onerror = () => reject(transaction.error);
    });
}

async function updateStats() {
    try {
        const data = await getDataFromDB();
        document.getElementById('tweetCount').textContent = data.tweets.length;
        document.getElementById('ratingCount').textContent = data.ratings.length;
    } catch (error) {
        console.error('Error updating stats:', error);
        document.getElementById('tweetCount').textContent = 'Error';
        document.getElementById('ratingCount').textContent = 'Error';
    }
}

// Export button
document.getElementById('exportData').addEventListener('click', async () => {
    try {
        const data = await getDataFromDB();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `tweet_data_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error exporting data:', error);
        alert('Error exporting data');
    }
});

// Clear button
document.getElementById('clearData').addEventListener('click', async () => {
    if (confirm('Are you sure you want to clear all data? This cannot be undone.')) {
        try {
            await clearDatabase();
            await updateStats();
            alert('All data cleared successfully!');
        } catch (error) {
            console.error('Error clearing data:', error);
            alert('Error clearing data');
        }
    }
});

// Update stats every second
setInterval(updateStats, 1000);

// Initial stats update
updateStats();