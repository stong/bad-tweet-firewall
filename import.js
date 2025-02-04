async function importJSONToIndexedDB(dbname, jsonData) {
    return new Promise((resolve, reject) => {
        try {
            // Validate input data
            if (!Array.isArray(jsonData)) {
                reject("Invalid JSON format: expected an array");
                return;
            }

            // Open IndexedDB
            const request = indexedDB.open("TweetDB", 1);

            request.onerror = () => reject("Error opening database");

            request.onsuccess = async (event) => {
                const db = event.target.result;
                const transaction = db.transaction(dbname, "readwrite");
                const store = transaction.objectStore(dbname);

                // Clear existing data
                store.clear();

                // Import all items
                let imported = 0;
                for (const item of jsonData) {
                    const addRequest = store.add(item);
                    addRequest.onsuccess = () => imported++;
                    addRequest.onerror = (e) => console.error("Error importing item:", e);
                }

                transaction.oncomplete = () => {
                    db.close();
                    resolve(imported);
                };

                transaction.onerror = () => {
                    db.close();
                    reject("Error during import transaction");
                };
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                // Create object store if it doesn't exist
                if (!db.objectStoreNames.contains(dbname)) {
                    db.createObjectStore(dbname, { keyPath: "id" });
                }
            };
        } catch (error) {
            reject(`Error processing data: ${error.message}`);
        }
    });
}

(() => {
alert("select tweets_export.json");

const input = document.createElement('input');
input.type = 'file';
input.onchange = e => {
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onload = e => {
        window.jsonData = JSON.parse(e.target.result);
        console.log('Data loaded into window.jsonData');
        importJSONToIndexedDB("tweets", window.jsonData)
            .then(count => console.log(`Imported ${count} items`))
            .catch(err => console.error(err));
    };
    reader.readAsText(file);
};
input.click();
})();


(() => {
alert("select ratings_export.json");

const input = document.createElement('input');
input.type = 'file';
input.onchange = e => {
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onload = e => {
        window.jsonData = JSON.parse(e.target.result);
        console.log('Data loaded into window.jsonData');
        importJSONToIndexedDB("ratings", window.jsonData)
            .then(count => console.log(`Imported ${count} items`))
            .catch(err => console.error(err));
    };
    reader.readAsText(file);
};
input.click();
})();