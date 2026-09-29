// Petrobras 3W Dashboard Interactive Client Script

let pressureChart = null;
let temperatureChart = null;
let labelChart = null;

let currentSeriesData = null;
let simInterval = null;
let simIndex = 0;

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    initCharts();
    await loadDatasetInfo();
    
    document.getElementById("class-select").addEventListener("change", onClassSelected);
    document.getElementById("btn-load-data").addEventListener("click", onLoadDataClicked);
    document.getElementById("btn-export-pdf").addEventListener("click", onExportPdfClicked);
    
    document.getElementById("btn-play").addEventListener("click", startSimulation);
    document.getElementById("btn-pause").addEventListener("click", pauseSimulation);
    document.getElementById("btn-reset").addEventListener("click", resetSimulation);
}

function onExportPdfClicked() {
    const classId = document.getElementById("class-select").value;
    const fileName = document.getElementById("file-select").value;
    if (!fileName) return;
    
    window.open(`/api/export_pdf_report/${classId}/${fileName}`, '_blank');
}

function playAlarmSound() {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5 note
        gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.3);
    } catch (e) {
        console.log("Audio Context blocked or not supported.");
    }
}

function initCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: { labels: { color: '#94A3B8', font: { family: 'Inter', size: 11 } } }
        },
        scales: {
            x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94A3B8', maxTicksLimit: 10 } },
            y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94A3B8' } }
        }
    };

    // 1. Pressure Chart
    const ctxP = document.getElementById("pressureChart").getContext("2d");
    pressureChart = new Chart(ctxP, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: chartOptions
    });

    // 2. Temperature Chart
    const ctxT = document.getElementById("temperatureChart").getContext("2d");
    temperatureChart = new Chart(ctxT, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: chartOptions
    });

    // 3. Label Chart
    const ctxL = document.getElementById("labelChart").getContext("2d");
    labelChart = new Chart(ctxL, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            ...chartOptions,
            scales: {
                ...chartOptions.scales,
                y: { min: -5, max: 120, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94A3B8' } }
            }
        }
    });
}

async function loadDatasetInfo() {
    try {
        const res = await fetch("/api/dataset_info");
        const data = await res.json();
        
        const classSelect = document.getElementById("class-select");
        classSelect.innerHTML = "";
        
        data.classes.forEach(c => {
            const opt = document.createElement("option");
            opt.value = c.class_id;
            opt.textContent = `${c.class_id}: ${c.name} (${c.count} dosya)`;
            classSelect.appendChild(opt);
        });
        
        if (data.model_info) {
            document.getElementById("model-name-display").textContent = `${data.model_info.model_name} (%${(data.model_info.accuracy * 100).toFixed(1)})`;
        }
        
        // Trigger first class selection
        if (data.classes.length > 0) {
            onClassSelected();
        }
    } catch (err) {
        console.error("Error loading dataset info:", err);
    }
}

async function onClassSelected() {
    const classId = document.getElementById("class-select").value;
    const fileSelect = document.getElementById("file-select");
    fileSelect.innerHTML = '<option value="">Yükleniyor...</option>';
    
    try {
        const res = await fetch(`/api/file_list/${classId}`);
        const data = await res.json();
        
        fileSelect.innerHTML = "";
        data.files.forEach(f => {
            const opt = document.createElement("option");
            opt.value = f;
            opt.textContent = f;
            fileSelect.appendChild(opt);
        });
        
        if (data.files.length > 0) {
            fileSelect.selectedIndex = 0;
        }
    } catch (err) {
        console.error("Error loading file list:", err);
    }
}

async function onLoadDataClicked() {
    const classId = document.getElementById("class-select").value;
    const fileName = document.getElementById("file-select").value;
    
    if (!fileName) return;
    
    resetSimulation();
    
    try {
        const res = await fetch(`/api/file_data/${classId}/${fileName}`);
        const data = await res.json();
        
        currentSeriesData = data;
        displayData(data);
    } catch (err) {
        console.error("Error loading file data:", err);
    }
}

function displayData(data) {
    document.getElementById("val-total-points").textContent = data.total_points.toLocaleString();
    
    // Calculate sample display metrics (last known value)
    const series = data.series;
    if (series['P-PDG']) {
        const lastVal = series['P-PDG'].filter(v => v !== null).slice(-1)[0];
        document.getElementById("val-p-pdg").innerHTML = lastVal ? `${(lastVal / 1e5).toFixed(1)} <small>bar</small>` : '--- <small>bar</small>';
    }
    if (series['T-TPT']) {
        const lastVal = series['T-TPT'].filter(v => v !== null).slice(-1)[0];
        document.getElementById("val-t-tpt").innerHTML = lastVal ? `${lastVal.toFixed(1)} <small>°C</small>` : '--- <small>°C</small>';
    }
    if (series['P-MON-CKP']) {
        const lastVal = series['P-MON-CKP'].filter(v => v !== null).slice(-1)[0];
        document.getElementById("val-p-mon-ckp").innerHTML = lastVal ? `${(lastVal / 1e5).toFixed(1)} <small>bar</small>` : '--- <small>bar</small>';
    }

    // 1. Update Pressure Chart
    const pDatasets = [];
    if (series['P-PDG']) pDatasets.push({ label: 'P-PDG (Kuyu Dibi)', data: series['P-PDG'].map(v => v ? v / 1e5 : null), borderColor: '#EF4444', borderWidth: 1.5, pointRadius: 0 });
    if (series['P-TPT']) pDatasets.push({ label: 'P-TPT (Ağaç Basıncı)', data: series['P-TPT'].map(v => v ? v / 1e5 : null), borderColor: '#3B82F6', borderWidth: 1.5, pointRadius: 0 });
    if (series['P-MON-CKP']) pDatasets.push({ label: 'P-MON-CKP (Şok Memba)', data: series['P-MON-CKP'].map(v => v ? v / 1e5 : null), borderColor: '#8B5CF6', borderWidth: 1.5, pointRadius: 0 });
    
    pressureChart.data.labels = data.timestamps;
    pressureChart.data.datasets = pDatasets;
    pressureChart.update();

    // 2. Update Temperature Chart
    const tDatasets = [];
    if (series['T-TPT']) tDatasets.push({ label: 'T-TPT (Ağaç Sıcaklığı)', data: series['T-TPT'], borderColor: '#10B981', borderWidth: 1.5, pointRadius: 0 });
    if (series['T-JUS-CKP']) tDatasets.push({ label: 'T-JUS-CKP (Şok Mansap)', data: series['T-JUS-CKP'], borderColor: '#F59E0B', borderWidth: 1.5, pointRadius: 0 });
    
    temperatureChart.data.labels = data.timestamps;
    temperatureChart.data.datasets = tDatasets;
    temperatureChart.update();

    // 3. Update Label Chart
    const lDatasets = [];
    if (series['class']) lDatasets.push({ label: 'Target Class Label', data: series['class'], borderColor: '#F43F5E', borderWidth: 2, pointRadius: 0, fill: true, backgroundColor: 'rgba(244, 63, 94, 0.1)' });
    
    labelChart.data.labels = data.timestamps;
    labelChart.data.datasets = lDatasets;
    labelChart.update();

    // 4. Update AI Prediction Box
    if (data.ai_prediction) {
        const pred = data.ai_prediction;
        document.getElementById("pred-empty").classList.add("hidden");
        document.getElementById("pred-result").classList.remove("hidden");
        
        const sevBadge = document.getElementById("pred-severity");
        sevBadge.textContent = pred.severity;
        sevBadge.style.backgroundColor = `${pred.color}25`;
        sevBadge.style.color = pred.color;
        sevBadge.style.border = `1px solid ${pred.color}50`;
        
        document.getElementById("pred-confidence").textContent = `%${pred.confidence} Güven`;
        document.getElementById("pred-class-name").textContent = pred.pred_class_name;
        document.getElementById("pred-desc").textContent = pred.desc;
        
        if (pred.severity === 'CRITICAL') {
            playAlarmSound();
        }
        
        const matchStatus = document.getElementById("pred-match-status");
        if (pred.is_match) {
            matchStatus.innerHTML = `<i class="fa-solid fa-circle-check"></i> Gerçek Sınıf Etiketi ile Doğru Eşleşti`;
            matchStatus.style.color = "#34D399";
        } else {
            matchStatus.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Sınıf Farklılık Gösteriyor`;
            matchStatus.style.color = "#FBBF24";
        }
    }
}

function startSimulation() {
    if (!currentSeriesData || simInterval) return;
    
    const totalPoints = currentSeriesData.timestamps.length;
    simInterval = setInterval(() => {
        simIndex += 5;
        if (simIndex >= totalPoints) {
            simIndex = totalPoints;
            pauseSimulation();
        }
        
        const pct = ((simIndex / totalPoints) * 100).toFixed(1);
        document.getElementById("sim-progress").style.width = `${pct}%`;
        
        // Slice data for simulation view
        const subData = {
            ...currentSeriesData,
            timestamps: currentSeriesData.timestamps.slice(0, simIndex),
            series: {}
        };
        
        for (let key in currentSeriesData.series) {
            subData.series[key] = currentSeriesData.series[key].slice(0, simIndex);
        }
        
        displayData(subData);
    }, 100);
}

function pauseSimulation() {
    if (simInterval) {
        clearInterval(simInterval);
        simInterval = null;
    }
}

function resetSimulation() {
    pauseSimulation();
    simIndex = 0;
    document.getElementById("sim-progress").style.width = "0%";
    if (currentSeriesData) {
        displayData(currentSeriesData);
    }
}
