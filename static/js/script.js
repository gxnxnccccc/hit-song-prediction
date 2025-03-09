// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Chart.js for feature importance
    initFeatureImportance();
    
    // Initialize sliders with value display
    initSliders();
    
    // Initialize prediction buttons
    document.getElementById('predict-custom').addEventListener('click', predictCustom);
    document.getElementById('predict-sample').addEventListener('click', predictSample);
    
    // Initialize song select change event
    document.getElementById('song-select').addEventListener('change', loadSongFeatures);
});

// Feature importance chart
let featureImportanceChart;

function initFeatureImportance() {
    // Fetch feature importance data from backend
    fetch('/feature_importance')
        .then(response => response.json())
        .then(data => {
            // Take only top 8 features for cleaner display
            const features = Object.keys(data).slice(0, 8);
            const values = features.map(feature => data[feature]);
            
            // Create chart
            const ctx = document.getElementById('feature-importance-chart').getContext('2d');
            featureImportanceChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: features.map(f => f.replace('_', ' ').replace(/^\w/, c => c.toUpperCase())),
                    datasets: [{
                        label: 'Importance',
                        data: values,
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y',
                    scales: {
                        x: {
                            beginAtZero: true
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        });
}

// Score gauge chart
let scoreGaugeChart;

function createScoreGauge(score) {
    // If chart exists, destroy it
    if (scoreGaugeChart) {
        scoreGaugeChart.destroy();
    }
    
    // Create new gauge chart
    const ctx = document.getElementById('score-gauge').getContext('2d');
    scoreGaugeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Score', 'Remaining'],
            datasets: [{
                data: [score, 100 - score],
                backgroundColor: [
                    getScoreColor(score),
                    'rgba(220, 220, 220, 0.2)'
                ],
                borderWidth: 0
            }]
        },
        options: {
            cutout: '75%',
            circumference: 270,
            rotation: -135,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                }
            }
        }
    });
}

// Get color based on score
function getScoreColor(score) {
    if (score >= 75) return 'rgba(40, 167, 69, 0.8)';  // High - green
    if (score >= 50) return 'rgba(255, 193, 7, 0.8)';  // Medium - yellow
    return 'rgba(220, 53, 69, 0.8)';  // Low - red
}

// Initialize all sliders
function initSliders() {
    document.querySelectorAll('.form-range').forEach(slider => {
        const valueDisplay = document.getElementById(`${slider.id}-value`);
        
        // Display formatted value
        valueDisplay.textContent = formatFeatureValue(slider.id, slider.value);
        
        // Update value on change
        slider.addEventListener('input', function() {
            valueDisplay.textContent = formatFeatureValue(this.id, this.value);
        });
    });
}

// Format feature values for display
function formatFeatureValue(feature, value) {
    value = parseFloat(value);
    
    // Format based on feature type
    if (feature === 'tempo') {
        return value.toFixed(1) + ' BPM';
    } else if (feature === 'loudness') {
        return value.toFixed(1) + ' dB';
    } else if (feature === 'duration_ms') {
        // Convert ms to MM:SS
        const minutes = Math.floor(value / 60000);
        const seconds = ((value % 60000) / 1000).toFixed(0);
        return `${minutes}:${seconds.padStart(2, '0')}`;
    } else if (feature === 'key') {
        // Convert numerical key to music notation
        const keyNames = ['C', 'C♯/D♭', 'D', 'D♯/E♭', 'E', 'F', 'F♯/G♭', 'G', 'G♯/A♭', 'A', 'A♯/B♭', 'B'];
        return keyNames[Math.round(value)];
    } else if (feature === 'mode') {
        return value == 1 ? 'Major' : 'Minor';
    } else if (feature === 'year') {
        return Math.round(value);
    } else {
        // Standard scaling features 0-1
        return value.toFixed(2);
    }
}

// Predict with custom features
function predictCustom() {
    // Collect all feature values
    const features = {};
    document.querySelectorAll('.form-range').forEach(slider => {
        features[slider.id] = parseFloat(slider.value);
    });
    
    // Send prediction request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features: features })
    })
    .then(response => response.json())
    .then(data => displayResults(data));
}

// Predict with sample song
function predictSample() {
    const songId = document.getElementById('song-select').value;
    
    if (!songId) {
        alert('Please select a song first');
        return;
    }
    
    // Send prediction request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ songId: songId })
    })
    .then(response => response.json())
    .then(data => displayResults(data));
}

// Load features for selected song
function loadSongFeatures() {
    const songId = document.getElementById('song-select').value;
    
    if (!songId) return;
    
    fetch(`/get_song_features/${songId}`)
        .then(response => response.json())
        .then(features => {
            // Update all sliders with song values
            for (const [feature, value] of Object.entries(features)) {
                const slider = document.getElementById(feature);
                if (slider) {
                    slider.value = value;
                    document.getElementById(`${feature}-value`).textContent = formatFeatureValue(feature, value);
                }
            }
        });
}

// Display prediction results
function displayResults(data) {
    // Show results card
    const resultsCard = document.getElementById('results-card');
    resultsCard.style.display = 'block';
    
    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Update results title if song info is available
    if (data.artist && data.song) {
        document.getElementById('results-title').textContent = `${data.artist} - ${data.song}`;
    } else {
        document.getElementById('results-title').textContent = 'Prediction Results';
    }
    
    // Update popularity score
    const score = Math.round(data.score);
    document.getElementById('popularity-score').textContent = score;
    
    // Create score gauge
    createScoreGauge(score);
    
    // Update hit verdict
    const hitVerdict = document.getElementById('hit-verdict');
    if (data.isHit) {
        hitVerdict.textContent = 'Potential Hit! 🎵';
        hitVerdict.className = 'mb-2 text-success';
    } else {
        hitVerdict.textContent = 'Needs Improvement';
        hitVerdict.className = 'mb-2 text-danger';
    }
    
    // Show actual popularity if available
    const actualPopularity = document.getElementById('actual-popularity');
    if (data.actual_popularity) {
        actualPopularity.textContent = `Actual popularity: ${data.actual_popularity}/100`;
        actualPopularity.style.display = 'block';
    } else {
        actualPopularity.style.display = 'none';
    }
    
    // Update factors
    const positiveFactors = document.getElementById('positive-factors');
    const negativeFactors = document.getElementById('negative-factors');
    
    // Clear previous factors
    positiveFactors.innerHTML = '';
    negativeFactors.innerHTML = '';
    
    // Add new factors
    data.positiveFactors.forEach(factor => {
        const li = document.createElement('li');
        li.className = 'list-group-item';
        li.textContent = factor;
        positiveFactors.appendChild(li);
    });
    
    data.negativeFactors.forEach(factor => {
        const li = document.createElement('li');
        li.className = 'list-group-item';
        li.textContent = factor;
        negativeFactors.appendChild(li);
    });
}