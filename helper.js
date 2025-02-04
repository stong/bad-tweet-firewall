async function exportTweetsToJSON(dbname, filename) {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open("TweetDB", 1);  // Explicitly set version 1
        
        request.onerror = () => reject("Error opening database");
        
        request.onsuccess = (event) => {
            const db = event.target.result;
            const transaction = db.transaction(dbname, "readonly");
            const store = transaction.objectStore(dbname);
            const tweets = [];
            
            store.openCursor().onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    tweets.push(cursor.value);
                    cursor.continue();
                } else {
                    // Create and download the JSON file
                    const blob = new Blob([JSON.stringify(tweets, null, 2)], {type: 'application/json'});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    a.click();
                    URL.revokeObjectURL(url);
                    
                    resolve(tweets);
                }
            };
            
            transaction.oncomplete = () => {
                db.close();
            };
        };
    });
}

// Run the export
exportTweetsToJSON("tweets", 'tweets_export.json')
    .then(tweets => console.log(`Exported ${tweets.length} tweets`))
    .catch(error => console.error('Export failed:', error));

exportTweetsToJSON("ratings", 'ratings_export.json')
    .then(tweets => console.log(`Exported ${tweets.length} tweets`))
    .catch(error => console.error('Export failed:', error));
