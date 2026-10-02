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
    document.getElementById("model-select").addEventListener("change", onModelSelected);
    document.getElementById("file-search-input").addEventListener("input", onSearchInput);
    document.getElementById("btn-load-data").addEventListener("click", onLoadDataClicked);
    document.getElementById("btn-export-pdf").addEventListener("click", onExportPdfClicked);
    
    const benchmarkBtn = document.getElementById("btn-open-benchmark");
    const modal = document.getElementById("benchmark-modal");
    const closeBtn = document.getElementById("btn-close-modal");
    
    if (benchmarkBtn && modal && closeBtn) {
        benchmarkBtn.addEventListener("click", () => modal.classList.remove("hidden"));
        closeBtn.addEventListener("click", () => modal.classList.add("hidden"));
        modal.addEventListener("click", (e) => {
            if (e.target === modal) modal.classList.add("hidden");
        });
    }

    // Fleet Monitor Modal Listeners
    const fleetBtn = document.getElementById("btn-open-fleet");
    const fleetModal = document.getElementById("fleet-modal");
    const closeFleetBtn = document.getElementById("btn-close-fleet-modal");
    if (fleetBtn && fleetModal && closeFleetBtn) {
        fleetBtn.addEventListener("click", () => {
            fleetModal.classList.remove("hidden");
            loadFleetStatus();
        });
        closeFleetBtn.addEventListener("click", () => fleetModal.classList.add("hidden"));
        fleetModal.addEventListener("click", (e) => {
            if (e.target === fleetModal) fleetModal.classList.add("hidden");
        });
    }

    // Upload & Retrain Modal Listeners
    const uploadBtn = document.getElementById("btn-open-upload");
    const uploadModal = document.getElementById("upload-modal");
    const closeUploadBtn = document.getElementById("btn-close-upload-modal");
    if (uploadBtn && uploadModal && closeUploadBtn) {
        uploadBtn.addEventListener("click", () => {
            uploadModal.classList.remove("hidden");
            populateUploadClasses();
        });
        closeUploadBtn.addEventListener("click", () => uploadModal.classList.add("hidden"));
        uploadModal.addEventListener("click", (e) => {
            if (e.target === uploadModal) uploadModal.classList.add("hidden");
        });
    }

    const uploadForm = document.getElementById("upload-form");
    if (uploadForm) {
        uploadForm.addEventListener("submit", onUploadSubmit);
    }

    const retrainBtn = document.getElementById("btn-trigger-retrain");
    if (retrainBtn) {
        retrainBtn.addEventListener("click", onRetrainClicked);
    }

    document.getElementById("btn-play").addEventListener("click", startSimulation);
    document.getElementById("btn-pause").addEventListener("click", pauseSimulation);
    document.getElementById("btn-reset").addEventListener("click", resetSimulation);

    // WebAR Modal Listeners
    const webarBtn = document.getElementById("btn-open-webar");
    const webarModal = document.getElementById("webar-modal");
    const closeWebarBtn = document.getElementById("btn-close-webar-modal");
    if (webarBtn && webarModal && closeWebarBtn) {
        webarBtn.addEventListener("click", () => webarModal.classList.remove("hidden"));
        closeWebarBtn.addEventListener("click", () => webarModal.classList.add("hidden"));
        webarModal.addEventListener("click", (e) => {
            if (e.target === webarModal) webarModal.classList.add("hidden");
        });
    }

    // Initialize 3D Christmas Tree Canvas
    setTimeout(init3DDigitalTwin, 200);
}


function onModelSelected() {
    const modelSelect = document.getElementById("model-select");
    const modelNameDisplay = document.getElementById("model-name-display");
    if (modelSelect.value === 'pytorch') {
        modelNameDisplay.textContent = "PyTorch 1D-CNN + BiLSTM";
    } else {
        modelNameDisplay.textContent = "XGBoost (92.35%)";
    }
    onLoadDataClicked();
}

let searchDebounceTimer = null;
function onSearchInput() {
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => {
        onClassSelected();
    }, 300);
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
    const searchQuery = document.getElementById("file-search-input").value;
    fileSelect.innerHTML = '<option value="">Yükleniyor...</option>';
    
    try {
        const url = `/api/file_list/${classId}?per_page=100&search=${encodeURIComponent(searchQuery)}`;
        const res = await fetch(url);
        const data = await res.json();
        
        fileSelect.innerHTML = "";
        if (data.files.length === 0) {
            fileSelect.innerHTML = '<option value="">Dosya bulunamadı</option>';
            return;
        }
        
        data.files.forEach(f => {
            const opt = document.createElement("option");
            opt.value = f;
            opt.textContent = f;
            fileSelect.appendChild(opt);
        });
        
        fileSelect.selectedIndex = 0;
    } catch (err) {
        console.error("Error loading file list:", err);
    }
}

async function onLoadDataClicked() {
    const classId = document.getElementById("class-select").value;
    const fileName = document.getElementById("file-select").value;
    const modelType = document.getElementById("model-select").value;
    
    if (!fileName) return;
    
    resetSimulation();
    
    try {
        const res = await fetch(`/api/file_data/${classId}/${fileName}?model_type=${modelType}`);
        const data = await res.json();
        
        currentSeriesData = data;
        displayData(data);
        loadShapExplain(classId, fileName);
    } catch (err) {
        console.error("Error loading file data:", err);
    }
}

async function loadShapExplain(classId, fileName) {
    const box = document.getElementById("shap-explain-box");
    const list = document.getElementById("shap-feature-list");
    
    try {
        const res = await fetch(`/api/shap_explain/${classId}/${fileName}`);
        const data = await res.json();
        
        if (data.top_features && data.top_features.length > 0) {
            box.classList.remove("hidden");
            list.innerHTML = "";
            data.top_features.forEach((item, idx) => {
                const el = document.createElement("div");
                el.style.display = "flex";
                el.style.justifyContent = "space-between";
                el.style.padding = "0.2rem 0";
                el.style.borderBottom = "1px solid rgba(255, 255, 255, 0.05)";
                el.innerHTML = `<span style="color: #94A3B8;">${idx + 1}. ${item.feature}</span> <b style="color: #34D399; font-family: monospace;">+${item.impact}</b>`;
                list.appendChild(el);
            });
            loadEngineeringRecommendations(classId, data.top_features);
        } else {
            box.classList.add("hidden");
            loadEngineeringRecommendations(classId, []);
        }
    } catch (e) {
        box.classList.add("hidden");
        loadEngineeringRecommendations(classId, []);
    }
}

async function loadEngineeringRecommendations(classId, topFeatures) {
    const box = document.getElementById("recommendation-box");
    const sevBadge = document.getElementById("rec-severity-badge");
    const titleEl = document.getElementById("rec-title");
    const stepsList = document.getElementById("rec-steps-list");
    const notesEl = document.getElementById("rec-sensor-notes");

    try {
        const topJson = topFeatures ? encodeURIComponent(JSON.stringify(topFeatures)) : '';
        const res = await fetch(`/api/recommendations/${classId}?top_features=${topJson}`);
        const data = await res.json();

        box.classList.remove("hidden");
        sevBadge.textContent = data.action_level || "RUTİN MÜDAHALE";
        sevBadge.style.backgroundColor = `${data.color}25`;
        sevBadge.style.color = data.color;
        sevBadge.style.border = `1px solid ${data.color}50`;

        titleEl.textContent = data.title;
        stepsList.innerHTML = "";
        data.steps.forEach(step => {
            const li = document.createElement("li");
            li.textContent = step;
            stepsList.appendChild(li);
        });

        notesEl.innerHTML = (data.sensor_notes && data.sensor_notes.length > 0) ? data.sensor_notes.join("<br>") : "";
    } catch (err) {
        console.error("Error loading recommendations:", err);
        box.classList.add("hidden");
    }
}

async function loadFleetStatus() {
    try {
        const res = await fetch("/api/fleet_status");
        const data = await res.json();

        const sum = data.fleet_summary;
        document.getElementById("fleet-health-score").textContent = `${sum.overall_health_score} / 100`;
        document.getElementById("fleet-normal-count").textContent = sum.normal_wells;
        document.getElementById("fleet-warning-count").textContent = sum.warning_wells;
        document.getElementById("fleet-critical-count").textContent = sum.critical_wells;

        const grid = document.getElementById("fleet-grid");
        grid.innerHTML = "";
        data.wells.forEach(w => {
            const card = document.createElement("div");
            card.style.background = "rgba(15, 23, 42, 0.7)";
            card.style.border = `1px solid ${w.color}50`;
            card.style.borderRadius = "10px";
            card.style.padding = "1rem";
            card.style.boxShadow = `0 4px 15px ${w.color}15`;

            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem;">${w.well_id}</span>
                    <span style="font-size: 0.7rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 6px; background: ${w.color}25; color: ${w.color}; border: 1px solid ${w.color}40;">${w.severity}</span>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.75rem;">${w.class_name}</div>
                <div style="font-size: 0.75rem; display: grid; grid-template-columns: 1fr 1fr; gap: 0.3rem; color: #CBD5E1;">
                    <div>P-PDG: <b style="color: #F8FAFC;">${w.p_pdg_bar} bar</b></div>
                    <div>T-TPT: <b style="color: #F8FAFC;">${w.t_tpt_c} °C</b></div>
                    <div>P-TPT: <b style="color: #F8FAFC;">${w.p_tpt_bar} bar</b></div>
                    <div>T-PDG: <b style="color: #F8FAFC;">${w.t_pdg_c} °C</b></div>
                </div>
            `;
            grid.appendChild(card);
        });
    } catch (err) {
        console.error("Error loading fleet status:", err);
    }
}

function populateUploadClasses() {
    const select = document.getElementById("upload-class-select");
    const origSelect = document.getElementById("class-select");
    if (select && origSelect) {
        select.innerHTML = origSelect.innerHTML;
    }
}

async function onUploadSubmit(e) {
    e.preventDefault();
    const fileInput = document.getElementById("upload-file-input");
    const classId = document.getElementById("upload-class-select").value;

    if (!fileInput.files || fileInput.files.length === 0) return;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("class_id", classId);

    const submitBtn = document.getElementById("btn-submit-upload");
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Yükleniyor...`;

    try {
        const res = await fetch("/api/upload_file", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        alert(`Başarılı: ${data.message}`);
        fileInput.value = "";
        onClassSelected();
    } catch (err) {
        alert(`Hata: ${err}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<i class="fa-solid fa-upload"></i> Veriyi Sisteme Yükle`;
    }
}

async function onRetrainClicked() {
    const btn = document.getElementById("btn-trigger-retrain");
    const statusDiv = document.getElementById("retrain-status");
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Model Yeniden Eğitiliyor / Güncelleniyor...`;
    statusDiv.textContent = "";

    try {
        const res = await fetch("/api/retrain_model", { method: "POST" });
        const data = await res.json();
        if (data.status === "Success") {
            statusDiv.style.color = "#10B981";
            statusDiv.textContent = `✓ Model Başarıyla Güncellendi! (Güncel Doğruluk: %${(data.accuracy * 100).toFixed(2)})`;
            await loadDatasetInfo();
        } else {
            statusDiv.style.color = "#EF4444";
            statusDiv.textContent = `Hata: ${data.error}`;
        }
    } catch (err) {
        statusDiv.style.color = "#EF4444";
        statusDiv.textContent = `Hata: ${err}`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-rotate"></i> Yapay Zeka Modelini Yeniden Eğit / Güncelle`;
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
        
        document.getElementById("pred-confidence").textContent = `%${pred.confidence} Güven (${pred.model_used || 'XGBoost'})`;
        document.getElementById("pred-class-name").textContent = pred.pred_class_name;
        document.getElementById("pred-desc").textContent = pred.desc;
        
        if (pred.severity === 'CRITICAL') {
            playAlarmSound();
        }
        
        // Update 3D Digital Twin Canvas
        update3DTwinStatus(pred.severity);
        
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

// ----------------------------------------------------
// 3D Digital Twin Viewport (Three.js WebGL Rendering)
// ----------------------------------------------------
let scene3D, camera3D, renderer3D, treeMesh3D, statusLight3D;
let isRotating3D = true;

function init3DDigitalTwin() {
    const container = document.getElementById("digital-twin-viewport");
    if (!container || typeof THREE === "undefined") return;

    scene3D = new THREE.Scene();
    scene3D.background = new THREE.Color(0x090d16);

    camera3D = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera3D.position.set(0, 3, 9);
    camera3D.lookAt(0, 1.2, 0);

    renderer3D = new THREE.WebGLRenderer({ antialias: true });
    renderer3D.setSize(container.clientWidth, container.clientHeight);
    renderer3D.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer3D.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene3D.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0x38bdf8, 1.4);
    dirLight.position.set(5, 10, 7);
    scene3D.add(dirLight);

    const group = new THREE.Group();

    // Main Pipe Column
    const colGeo = new THREE.CylinderGeometry(0.35, 0.35, 3.5, 16);
    const colMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.8, roughness: 0.2 });
    const colMesh = new THREE.Mesh(colGeo, colMat);
    colMesh.position.y = 1.75;
    group.add(colMesh);

    // Cross Wing Pipes
    const pipeGeo = new THREE.CylinderGeometry(0.22, 0.22, 3.2, 16);
    const pipeMesh = new THREE.Mesh(pipeGeo, colMat);
    pipeMesh.rotation.z = Math.PI / 2;
    pipeMesh.position.y = 2.2;
    group.add(pipeMesh);

    // Choke Valve (Spherical glowing indicator)
    const valveGeo = new THREE.SphereGeometry(0.45, 16, 16);
    const valveMat = new THREE.MeshStandardMaterial({ color: 0x10b981, emissive: 0x047857, roughness: 0.3 });
    statusLight3D = new THREE.Mesh(valveGeo, valveMat);
    statusLight3D.position.set(1.5, 2.2, 0);
    group.add(statusLight3D);

    // Base Flange
    const baseGeo = new THREE.BoxGeometry(2.6, 0.3, 2.6);
    const baseMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.9 });
    const baseMesh = new THREE.Mesh(baseGeo, baseMat);
    baseMesh.position.y = 0.15;
    group.add(baseMesh);

    scene3D.add(group);
    treeMesh3D = group;

    function renderLoop() {
        requestAnimationFrame(renderLoop);
        if (isRotating3D && treeMesh3D) {
            treeMesh3D.rotation.y += 0.008;
        }
        renderer3D.render(scene3D, camera3D);
    }
    renderLoop();

    const rotateBtn = document.getElementById("btn-toggle-3d-rotate");
    if (rotateBtn) {
        rotateBtn.addEventListener("click", () => { isRotating3D = !isRotating3D; });
    }
}

function update3DTwinStatus(severity) {
    if (!statusLight3D) return;
    const dot = document.getElementById("twin-status-dot");
    const txt = document.getElementById("twin-status-text");
    if (severity === "CRITICAL") {
        statusLight3D.material.color.setHex(0xef4444);
        statusLight3D.material.emissive.setHex(0xb91c1c);
        if (dot) dot.style.color = "#EF4444";
        if (txt) txt.textContent = "CRITICAL FAULT";
    } else if (severity === "WARNING") {
        statusLight3D.material.color.setHex(0xf59e0b);
        statusLight3D.material.emissive.setHex(0xb45309);
        if (dot) dot.style.color = "#F59E0B";
        if (txt) txt.textContent = "WARNING";
    } else {
        statusLight3D.material.color.setHex(0x10b981);
        statusLight3D.material.emissive.setHex(0x047857);
        if (dot) dot.style.color = "#10B981";
        if (txt) txt.textContent = "NORMAL";
    }
}

