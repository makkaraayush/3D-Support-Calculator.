/**
 * Frontend Controller for 3D Support Calculator & AI Engine.
 * Manages model uploads, REST API communication, continuous learning SSE stream,
 * slider controls, and Bambu Lab 3MF exports.
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize 3D Viewport
  const viewer = new SupportViewer3D('threeViewport');

  // DOM Elements
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const browseFileBtn = document.getElementById('browseFileBtn');
  const benchmarkSelect = document.getElementById('benchmarkSelect');

  const materialSelect = document.getElementById('materialSelect');
  const layerHeightSelect = document.getElementById('layerHeightSelect');
  const prioritySelect = document.getElementById('prioritySelect');

  const lblAirGap = document.getElementById('lblAirGap');
  const lblSafeAngle = document.getElementById('lblSafeAngle');
  const lblMaxBridge = document.getElementById('lblMaxBridge');

  const sliderBranchAngle = document.getElementById('sliderBranchAngle');
  const sliderMergeDist = document.getElementById('sliderMergeDist');
  const sliderTipDia = document.getElementById('sliderTipDia');
  const sliderGridRes = document.getElementById('sliderGridRes');

  const valBranchAngle = document.getElementById('valBranchAngle');
  const valMergeDist = document.getElementById('valMergeDist');
  const valTipDia = document.getElementById('valTipDia');
  const valGridRes = document.getElementById('valGridRes');

  const generateSupportBtn = document.getElementById('generateSupportBtn');
  const resetParamsBtn = document.getElementById('resetParamsBtn');

  // Continuous Engine Elements
  const btnStartEngine = document.getElementById('btnStartEngine');
  const btnStopEngine = document.getElementById('btnStopEngine');
  const engineStatusBadge = document.getElementById('engineStatusBadge');
  const engineStatusText = document.getElementById('engineStatusText');
  const lblGenCount = document.getElementById('lblGenCount');
  const lblSurrogateLoss = document.getElementById('lblSurrogateLoss');
  const lblBestFitness = document.getElementById('lblBestFitness');
  const lblFilamentSaved = document.getElementById('lblFilamentSaved');
  const lblActivityTicker = document.getElementById('lblActivityTicker');

  // Viewport Toolbar Buttons
  const btnToggleModel = document.getElementById('btnToggleModel');
  const btnToggleHeatmap = document.getElementById('btnToggleHeatmap');
  const btnToggleSupport = document.getElementById('btnToggleSupport');
  const btnToggleBed = document.getElementById('btnToggleBed');
  const btnResetCamera = document.getElementById('btnResetCamera');

  // Export Buttons
  const exportStlBtn = document.getElementById('exportStlBtn');
  const export3mfBtn = document.getElementById('export3mfBtn');

  let sseEventSource = null;

  // 2. Load Presets and Benchmarks
  async function loadInitialData() {
    try {
      // Presets
      const resPresets = await fetch('/api/presets');
      const presets = await resPresets.json();
      window.lastPresets = presets;
      updatePhysicsCallout(presets);

      // Benchmarks
      const resBench = await fetch('/api/benchmarks');
      const benchData = await resBench.json();
      benchmarkSelect.innerHTML = '<option value="">Select benchmark model...</option>';
      benchData.models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.name;
        benchmarkSelect.appendChild(opt);
      });

      // Automatically load the first benchmark model on launch
      if (benchData.models.length > 0) {
        benchmarkSelect.value = benchData.models[0].id;
        loadBenchmarkModel(benchData.models[0].id);
      }
    } catch (err) {
      console.error('Failed to load initial presets:', err);
    }
  }

  function updatePhysicsCallout(presets) {
    const matName = materialSelect.value;
    const lHeight = parseFloat(layerHeightSelect.value);
    if (!presets || !presets.materials) return;

    const matInfo = presets.materials[matName];
    if (matInfo) {
      lblSafeAngle.textContent = `${matInfo.safe_overhang_base_deg}°`;
      lblAirGap.textContent = `${matInfo.optimal_z_gap_020} mm`;
      lblMaxBridge.textContent = matName === 'PLA' ? '32 mm' : (matName === 'PETG' ? '22 mm' : '26 mm');
    }
  }

  // 3. File Upload and Benchmark Handlers
  browseFileBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) uploadFile(e.target.files[0]);
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
  });

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('material', materialSelect.value);
    formData.append('layer_height', layerHeightSelect.value);
    formData.append('priority', prioritySelect.value);

    lblActivityTicker.textContent = `Uploading ${file.name}...`;

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');
      updateModelSpecs(data);
      await refresh3DGeometry();
      lblActivityTicker.textContent = `Loaded ${file.name} successfully.`;
      // Trigger instant support generation
      triggerSupportGeneration();
    } catch (err) {
      alert(`Error loading file: ${err.message}`);
      lblActivityTicker.textContent = 'Upload failed';
    }
  }

  benchmarkSelect.addEventListener('change', (e) => {
    if (e.target.value) loadBenchmarkModel(e.target.value);
  });

  async function loadBenchmarkModel(modelId) {
    lblActivityTicker.textContent = `Loading benchmark ${modelId}...`;
    try {
      const res = await fetch(`/api/benchmarks/load/${modelId}`, { method: 'POST' });
      const data = await res.json();
      updateModelSpecs(data);
      await refresh3DGeometry();
      lblActivityTicker.textContent = `Benchmark ${modelId} loaded.`;
      triggerSupportGeneration();
    } catch (err) {
      console.error(err);
    }
  }

  function updateModelSpecs(data) {
    document.getElementById('specModelName').textContent = data.filename || 'Custom Model';
    document.getElementById('specTriangles').textContent = (data.face_count || 0).toLocaleString();
    if (data.dimensions) {
      document.getElementById('specDimensions').textContent = `${data.dimensions[0]} × ${data.dimensions[1]} × ${data.dimensions[2]} mm`;
    }
    if (data.overhang_analysis) {
      document.getElementById('specOverhangPct').textContent = `${data.overhang_analysis.critical_overhang_pct}% (${data.overhang_analysis.critical_face_count} faces)`;
    }
  }

  async function refresh3DGeometry() {
    try {
      const res = await fetch('/api/mesh/current-geometry');
      const geoData = await res.json();
      viewer.loadGeometry(geoData);
    } catch (err) {
      console.error('Failed to refresh 3D geometry:', err);
    }
  }

  // 4. Interactive Sliders
  sliderBranchAngle.addEventListener('input', (e) => valBranchAngle.textContent = `${e.target.value}°`);
  sliderMergeDist.addEventListener('input', (e) => valMergeDist.textContent = `${parseFloat(e.target.value).toFixed(1)} mm`);
  sliderTipDia.addEventListener('input', (e) => valTipDia.textContent = `${parseFloat(e.target.value).toFixed(2)} mm`);
  sliderGridRes.addEventListener('input', (e) => valGridRes.textContent = `${parseFloat(e.target.value).toFixed(1)} mm`);

  materialSelect.addEventListener('change', () => {
    updatePhysicsCallout(window.lastPresets);
    triggerSupportGeneration();
  });
  layerHeightSelect.addEventListener('change', () => {
    updatePhysicsCallout(window.lastPresets);
    triggerSupportGeneration();
  });
  prioritySelect.addEventListener('change', () => {
    triggerSupportGeneration();
  });

  resetParamsBtn.addEventListener('click', () => {
    sliderBranchAngle.value = 28; valBranchAngle.textContent = '28°';
    sliderMergeDist.value = 9.0; valMergeDist.textContent = '9.0 mm';
    sliderTipDia.value = 0.70; valTipDia.textContent = '0.70 mm';
    sliderGridRes.value = 3.4; valGridRes.textContent = '3.4 mm';
    triggerSupportGeneration();
  });

  generateSupportBtn.addEventListener('click', () => triggerSupportGeneration());

  async function triggerSupportGeneration() {
    lblActivityTicker.textContent = 'Simulating tree support physics...';
    const params = new URLSearchParams({
      branch_angle_deg: sliderBranchAngle.value,
      merge_distance_mm: sliderMergeDist.value,
      tip_diameter_mm: sliderTipDia.value,
      grid_res_mm: sliderGridRes.value,
      material: materialSelect.value,
      layer_height: layerHeightSelect.value,
      priority: prioritySelect.value,
    });

    try {
      const res = await fetch(`/api/generate?${params.toString()}`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Generation failed');

      updatePhysicsReport(data.report);
      await refresh3DGeometry();

      if (data.tips_count === 0) {
        lblActivityTicker.textContent = 'Model is 100% self-supporting. No supports needed!';
      } else {
        lblActivityTicker.textContent = `Generated ${data.tips_count} elevated support tips. Saved ${data.report.filament_saved_vs_grid_pct}% filament!`;
      }
    } catch (err) {
      console.error(err);
      lblActivityTicker.textContent = 'Generation error';
    }
  }

  function updatePhysicsReport(rep) {
    if (!rep) return;
    // Gauges
    document.getElementById('repCoverage').textContent = `${rep.coverage_score_pct}%`;
    document.getElementById('barCoverage').style.width = `${Math.min(100, rep.coverage_score_pct)}%`;

    document.getElementById('repRemovalEase').textContent = `${rep.removal_ease_score} / 10 (${rep.surface_scarring_risk})`;
    document.getElementById('barRemovalEase').style.width = `${rep.removal_ease_score * 10}%`;

    document.getElementById('repBuckling').textContent = `${rep.min_buckling_safety_factor}x`;
    document.getElementById('barBuckling').style.width = `${Math.min(100, rep.min_buckling_safety_factor * 12)}%`;

    document.getElementById('repBedAdhesion').textContent = `${rep.bed_adhesion_safety_factor}x`;
    document.getElementById('barBedAdhesion').style.width = `${Math.min(100, rep.bed_adhesion_safety_factor * 8)}%`;

    // Benchmark Table
    const formatTime = (mins) => {
      const m = Math.round(mins || 0);
      if (m < 60) return `${m}m`;
      const h = Math.floor(m / 60);
      const rem = m % 60;
      return `${h}h ${rem}m`;
    };

    const aiMass = rep.support_mass_grams;
    const gridMass = rep.grid_mass_grams ? rep.grid_mass_grams.toFixed(1) : (aiMass / Math.max(0.1, (1.0 - (rep.filament_saved_vs_grid_pct / 100.0)))).toFixed(1);
    const treeMass = rep.tree_mass_grams ? rep.tree_mass_grams.toFixed(1) : (aiMass / Math.max(0.1, (1.0 - (rep.filament_saved_vs_tree_pct / 100.0)))).toFixed(1);

    document.getElementById('cmpGridFilament').textContent = `${gridMass} g`;
    document.getElementById('cmpGridTime').textContent = formatTime(rep.grid_print_time_minutes || (rep.print_time_minutes * 1.5));

    document.getElementById('cmpTreeFilament').textContent = `${treeMass} g`;
    document.getElementById('cmpTreeTime').textContent = formatTime(rep.standard_tree_print_time_minutes || (rep.print_time_minutes * 1.2));

    document.getElementById('cmpAIFilament').textContent = `${aiMass.toFixed(1)} g (-${rep.filament_saved_vs_grid_pct}%)`;
    document.getElementById('cmpAITime').textContent = formatTime(rep.print_time_minutes);
  }

  // 5. Continuous Self-Learning Engine SSE Stream
  btnStartEngine.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/optimizer/start', { method: 'POST' });
      const data = await res.json();
      if (data.is_running) {
        setEngineRunningUI(true);
        startSSEStream();
      }
    } catch (err) {
      console.error(err);
    }
  });

  btnStopEngine.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/optimizer/stop', { method: 'POST' });
      setEngineRunningUI(false);
      if (sseEventSource) sseEventSource.close();
    } catch (err) {
      console.error(err);
    }
  });

  function setEngineRunningUI(isRunning) {
    if (isRunning) {
      btnStartEngine.classList.add('hidden');
      btnStopEngine.classList.remove('hidden');
      engineStatusBadge.classList.add('active');
      engineStatusText.textContent = 'AI Engine: Exploring Possibilities';
    } else {
      btnStartEngine.classList.remove('hidden');
      btnStopEngine.classList.add('hidden');
      engineStatusBadge.classList.remove('active');
      engineStatusText.textContent = 'AI Engine: Paused';
    }
  }

  function startSSEStream() {
    if (sseEventSource) sseEventSource.close();
    sseEventSource = new EventSource('/api/optimizer/stream');

    sseEventSource.onmessage = (event) => {
      try {
        const state = JSON.parse(event.data);
        lblGenCount.textContent = state.total_iterations;
        lblSurrogateLoss.textContent = state.loss.toFixed(4);
        lblBestFitness.textContent = state.best_fitness.toFixed(1);
        lblFilamentSaved.textContent = `${state.filament_saved_pct}%`;
        lblActivityTicker.textContent = state.status_message;

        // If optimizer updated best parameters, sync sliders
        if (state.best_parameters) {
          const bp = state.best_parameters;
          sliderBranchAngle.value = Math.round(bp.branch_angle_deg);
          valBranchAngle.textContent = `${Math.round(bp.branch_angle_deg)}°`;

          sliderMergeDist.value = bp.merge_distance_mm.toFixed(1);
          valMergeDist.textContent = `${bp.merge_distance_mm.toFixed(1)} mm`;

          sliderTipDia.value = bp.tip_diameter_mm.toFixed(2);
          valTipDia.textContent = `${bp.tip_diameter_mm.toFixed(2)} mm`;

          sliderGridRes.value = bp.contact_grid_res_mm.toFixed(1);
          valGridRes.textContent = `${bp.contact_grid_res_mm.toFixed(1)} mm`;
        }

        // Periodically refresh 3D view with the top candidate
        if (state.total_iterations % 8 === 0) {
          refresh3DGeometry();
        }
      } catch (err) {
        console.error('Error parsing SSE event:', err);
      }
    };

    sseEventSource.onerror = () => {
      console.warn('SSE connection closed or lost.');
    };
  }

  // 6. Viewport Toolbar Toggles
  btnToggleModel.addEventListener('click', () => {
    const isVis = viewer.toggleModel();
    btnToggleModel.classList.toggle('active', isVis);
  });
  btnToggleHeatmap.addEventListener('click', () => {
    const isHeat = viewer.toggleHeatmap();
    btnToggleHeatmap.classList.toggle('active', isHeat);
  });
  btnToggleSupport.addEventListener('click', () => {
    const isSupp = viewer.toggleSupport();
    btnToggleSupport.classList.toggle('active', isSupp);
  });
  btnToggleBed.addEventListener('click', () => {
    const isBed = viewer.toggleBed();
    btnToggleBed.classList.toggle('active', isBed);
  });
  btnResetCamera.addEventListener('click', () => viewer.resetView());

  // 7. Exports
  exportStlBtn.addEventListener('click', () => {
    window.location.href = '/api/export/stl?include_model=true';
  });
  export3mfBtn.addEventListener('click', () => {
    window.location.href = '/api/export/3mf';
  });

  // Launch initial setup
  loadInitialData();
});
