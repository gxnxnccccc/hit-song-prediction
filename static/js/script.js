document.addEventListener('DOMContentLoaded', function() {
    
    initFeatureImportance();
    
    initSliders();
    
    document.getElementById('predict-custom').addEventListener('click', predictCustom);
    document.getElementById('predict-sample').addEventListener('click', predictSample);
    
    document.getElementById('song-select').addEventListener('change', loadSongFeatures);
});

let featureImportanceChart;

function initFeatureImportance() {

    fetch('/feature_importance')
        .then(response => response.json())
        .then(data => {
            const features = Object.keys(data).slice(0, 8);
            const values = features.map(feature => data[feature]);
            
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

let scoreGaugeChart;

function createScoreGauge(score) {

    if (scoreGaugeChart) {
        scoreGaugeChart.destroy();
    }
    
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

function getScoreColor(score) {
    if (score >= 75) return 'rgba(40, 167, 69, 0.8)';  // High - green
    if (score >= 50) return 'rgba(255, 193, 7, 0.8)';  // Medium - yellow
    return 'rgba(220, 53, 69, 0.8)';  // Low - red
}

function initSliders() {
    document.querySelectorAll('.form-range').forEach(slider => {
        const valueDisplay = document.getElementById(`${slider.id}-value`);
        
        valueDisplay.textContent = formatFeatureValue(slider.id, slider.value);
        
        slider.addEventListener('input', function() {
            valueDisplay.textContent = formatFeatureValue(this.id, this.value);
        });
    });
}

function formatFeatureValue(feature, value) {
    value = parseFloat(value);
    
    if (feature === 'tempo') {
        return value.toFixed(1) + ' BPM';
    } else if (feature === 'loudness') {
        return value.toFixed(1) + ' dB';
    } else if (feature === 'duration_ms') {
        const minutes = Math.floor(value / 60000);
        const seconds = ((value % 60000) / 1000).toFixed(0);
        return `${minutes}:${seconds.padStart(2, '0')}`;
    } else if (feature === 'key') {
        const keyNames = ['C', 'C♯/D♭', 'D', 'D♯/E♭', 'E', 'F', 'F♯/G♭', 'G', 'G♯/A♭', 'A', 'A♯/B♭', 'B'];
        return keyNames[Math.round(value)];
    } else if (feature === 'mode') {
        return value == 1 ? 'Major' : 'Minor';
    } else if (feature === 'year') {
        return Math.round(value);
    } else {
        return value.toFixed(2);
    }
}

function displayFeatureValues(features) {
    const summaryContainer = document.getElementById('feature-values-summary');
    summaryContainer.innerHTML = ''; 
    
    fetch('/feature_importance')
        .then(response => response.json())
        .then(importance => {
            let featuresList = Object.keys(features);
            featuresList.sort((a, b) => {
                if (importance[a] !== undefined && importance[b] !== undefined) {
                    return importance[b] - importance[a];
                }

                return 0;
            });
            
            for (const feature of featuresList) {
                const value = features[feature];
                const formattedValue = formatFeatureValue(feature, value);
                
                const col = document.createElement('div');
                col.className = 'col-md-4 mb-3';
                
                const featureDisplay = document.createElement('div');
                featureDisplay.className = 'feature-summary-item';
                
                if (importance[feature] !== undefined) {
                    const importanceIndicator = document.createElement('div');
                    importanceIndicator.className = 'importance-indicator';
                    const importanceWidth = Math.min(100, Math.round(importance[feature] * 100 / 0.1));
                    importanceIndicator.innerHTML = `
                        <div class="progress" style="height: 4px;">
                            <div class="progress-bar bg-info" role="progressbar" 
                                style="width: ${importanceWidth}%" 
                                aria-valuenow="${importanceWidth}" aria-valuemin="0" aria-valuemax="100"></div>
                        </div>
                    `;
                    featureDisplay.appendChild(importanceIndicator);
                }
                
                const featureName = document.createElement('div');
                featureName.className = 'feature-name';
                featureName.textContent = feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                
                const featureValue = document.createElement('div');
                featureValue.className = 'feature-value-display';
                featureValue.textContent = formattedValue;
                
                featureDisplay.appendChild(featureName);
                featureDisplay.appendChild(featureValue);
                col.appendChild(featureDisplay);
                summaryContainer.appendChild(col);
            }
        })
        .catch(() => {
            let featuresList = Object.keys(features);
            
            for (const feature of featuresList) {
                const value = features[feature];
                const formattedValue = formatFeatureValue(feature, value);
                
                const col = document.createElement('div');
                col.className = 'col-md-4 mb-3';
                
                const featureDisplay = document.createElement('div');
                featureDisplay.className = 'feature-summary-item';
                
                const featureName = document.createElement('div');
                featureName.className = 'feature-name';
                featureName.textContent = feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                
                const featureValue = document.createElement('div');
                featureValue.className = 'feature-value-display';
                featureValue.textContent = formattedValue;
                
                featureDisplay.appendChild(featureName);
                featureDisplay.appendChild(featureValue);
                col.appendChild(featureDisplay);
                summaryContainer.appendChild(col);
            }
        });
}

function predictCustom() {
    const features = {};
    document.querySelectorAll('.form-range').forEach(slider => {
        features[slider.id] = parseFloat(slider.value);
    });
    
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

function predictSample() {
    const songId = document.getElementById('song-select').value;
    
    if (!songId) {
        alert('Please select a song first');
        return;
    }
    
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

function loadSongFeatures() {
    const songId = document.getElementById('song-select').value;
    
    if (!songId) return;
    
    fetch(`/get_song_features/${songId}`)
        .then(response => response.json())
        .then(features => {
            for (const [feature, value] of Object.entries(features)) {
                const slider = document.getElementById(feature);
                if (slider) {
                    slider.value = value;
                    document.getElementById(`${feature}-value`).textContent = formatFeatureValue(feature, value);
                }
            }
        });
}

function displayResults(data) {
    const resultsCard = document.getElementById('results-card');
    resultsCard.style.display = 'block';
    
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    if (data.artist && data.song) {
        document.getElementById('results-title').textContent = `${data.artist} - ${data.song}`;
    } else {
        document.getElementById('results-title').textContent = 'Prediction Results';
    }
    
    const score = Math.round(data.score);
    document.getElementById('popularity-score').textContent = score;
    
    createScoreGauge(score);
    
    const hitVerdict = document.getElementById('hit-verdict');
    if (data.isHit) {
        hitVerdict.textContent = 'Potential Hit! 🎵';
        hitVerdict.className = 'mb-2 text-success';
    } else {
        // hitVerdict.textContent = 'Needs Improvement';
        hitVerdict.className = 'mb-2 text-danger';
    }

    const actualPopularity = document.getElementById('actual-popularity');
    if (data.actual_popularity) {
        actualPopularity.textContent = `Actual popularity: ${data.actual_popularity}/100`;
        actualPopularity.style.display = 'block';
    } else {
        actualPopularity.style.display = 'none';
    }
    
    const positiveFactors = document.getElementById('positive-factors');
    const negativeFactors = document.getElementById('negative-factors');
    
    positiveFactors.innerHTML = '';
    negativeFactors.innerHTML = '';
    
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

    if (data.features) {
        displayFeatureValues(data.features);
    }
}

