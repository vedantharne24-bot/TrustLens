let currentMode = 'text';

function setMode(mode) {
    currentMode = mode;
    document.getElementById('textBtn').classList.toggle('active', mode === 'text');
    document.getElementById('linkBtn').classList.toggle('active', mode === 'link');
    document.getElementById('reviewInput').placeholder = 
        mode === 'text' ? "Paste review text..." : "Paste product URL...";
}

async function scanReview() {
    const inputField = document.getElementById('reviewInput');
    const list = document.getElementById('resultsList');
    const btnText = document.getElementById('btnText');
    
    if (!inputField.value.trim()) return alert("Input required!");
    
    btnText.innerText = "CRUNCHING NEURONS...";
    list.innerHTML = ''; // Clear old results

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                text: inputField.value, 
                is_link: currentMode === 'link' 
            })
        });
        const data = await response.json();

        if (data.results) {
            data.results.forEach(res => {
                const color = res.prediction === 'Fake' ? '#ff4d4d' : '#00ff88';
                const card = document.createElement('div');
                card.className = 'result-item';
                card.innerHTML = `
                    <div style="display:flex; justify-content:space-between">
                        <h4 style="color:${color}">${res.prediction.toUpperCase()}</h4>
                        <strong>${res.confidence}% Confidence</strong>
                    </div>
                    <p style="font-size:0.9rem; margin-top:10px; color:#ccc">"${res.text}"</p>
                    <div class="track"><div class="fill" style="width:${res.confidence}%; background:${color}"></div></div>
                `;
                list.appendChild(card); // Injects new card into the grid
            });
        }
    } catch (err) {
        alert("Neural link severed. Ensure app.py is online.");
    } finally {
        btnText.innerText = "INITIALIZE SCAN";
    }
}