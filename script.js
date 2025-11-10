    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const statusEl = document.getElementById("status");
    const ctx = canvas.getContext("2d");
    let detector; // nuevo detector que soporta múltiples manos
  // Dataset en memoria: { label: [featureVector, ...] }
  let dataset = {};
    const maxNeighbors = 5;

    const addSampleBtn = document.getElementById('addSample');
    const clearSamplesBtn = document.getElementById('clearSamples');
    const letterSelect = document.getElementById('letter');
    const datasetInfo = document.getElementById('datasetInfo');
    const predictionEl = document.getElementById('prediction');
    const exportBtn = document.getElementById('exportBtn');
    const importBtn = document.getElementById('importBtn');
    const importFile = document.getElementById('importFile');
    const autoCountInput = document.getElementById('autoCount');
    const autoIntervalInput = document.getElementById('autoInterval');
    const autoCollectBtn = document.getElementById('autoCollectBtn');
    const smoothSelect = document.getElementById('smooth');
    const predictionBig = document.getElementById('predictionBig');    // smoothing buffer
    let predBuffer = [];
    function pushPrediction(p) {
      const windowSize = parseInt(smoothSelect.value || '5', 10) || 5;
      predBuffer.push(p || null);
      if (predBuffer.length > windowSize) predBuffer.shift();
      // majority vote
      const counts = {};
      for (const v of predBuffer) if (v) counts[v] = (counts[v] || 0) + 1;
      let best = null, bestCount = 0;
      for (const k of Object.keys(counts)) {
        if (counts[k] > bestCount) { best = k; bestCount = counts[k]; }
      }
      predictionEl.textContent = `Predicción: ${best || '-'}`;
      predictionBig.textContent = best || '-';
    }

    // localStorage autosave key
    const STORAGE_KEY = 'signDatasetV1';
    function saveDatasetToLocalStorage() {
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(dataset)); } catch (e) { console.warn('No se pudo guardar dataset:', e); }
    }
    function loadDatasetFromLocalStorage() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === 'object') {
            dataset = parsed;
            updateDatasetInfo();
          }
        }
      } catch (e) { console.warn('No se pudo cargar dataset:', e); }
    }

    // export / import handlers
    exportBtn.addEventListener('click', () => {
      const dataStr = JSON.stringify(dataset);
      const blob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'sign-dataset.json';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });

    importBtn.addEventListener('click', () => importFile.click());
    importFile.addEventListener('change', (ev) => {
      const f = ev.target.files && ev.target.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const parsed = JSON.parse(reader.result);
          if (parsed && typeof parsed === 'object') {
            dataset = parsed;
            updateDatasetInfo();
            saveDatasetToLocalStorage();
            statusEl.textContent = 'Dataset importado correctamente.';
          }
        } catch (e) { statusEl.textContent = 'Error al importar JSON.'; }
      };
      reader.readAsText(f);
    });

    // Auto-collect
    autoCollectBtn.addEventListener('click', async () => {
      const count = Math.max(1, parseInt(autoCountInput.value || '10', 10));
      const interval = Math.max(50, parseInt(autoIntervalInput.value || '200', 10));
      const label = letterSelect.value;
      statusEl.textContent = `Auto-collect: intentando capturar ${count} muestras...`;
      let collected = 0;
      for (let i = 0; i < count; i++) {
        await new Promise(r => setTimeout(r, interval));
        if (lastKeypoints) {
          const fv = keypointsToFeature(lastKeypoints);
          if (!dataset[label]) dataset[label] = [];
          dataset[label].push(fv);
          collected++;
          updateDatasetInfo();
        }
      }
      saveDatasetToLocalStorage();
      statusEl.textContent = `Auto-collect finalizado: ${collected}/${count} añadidas.`;
    });

    // cargar dataset guardado al inicio
    loadDatasetFromLocalStorage();

    addSampleBtn.addEventListener('click', () => {
      if (!lastKeypoints) { statusEl.textContent = 'No hay keypoints para guardar.'; return; }
      const label = letterSelect.value;
      const fv = keypointsToFeature(lastKeypoints);
      if (!dataset[label]) dataset[label] = [];
      dataset[label].push(fv);
      updateDatasetInfo();
    });

    clearSamplesBtn.addEventListener('click', () => {
      for (const k of Object.keys(dataset)) delete dataset[k];
      updateDatasetInfo();
    });

    function updateDatasetInfo() {
      let total = 0;
      for (const k of Object.keys(dataset)) total += dataset[k].length;
      datasetInfo.textContent = `Muestras: ${total}`;
    }

    function keypointsToFeature(keypoints) {
      // Normalizar: usar el bounding box de la mano para tener invarianza a escala
      const pts = keypoints.map(p => [p.x, p.y]);
      // calcular bbox
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const [x, y] of pts) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
      const w = Math.max(1e-6, maxX - minX);
      const h = Math.max(1e-6, maxY - minY);
      // Center and scale
      const fv = [];
      for (const [x, y] of pts) {
        fv.push((x - minX) / w);
        fv.push((y - minY) / h);
      }
      return fv; // length 42 (21*2)
    }

    function knnPredict(fv) {
      // Construir lista de [dist, label]
      const neighbors = [];
      for (const label of Object.keys(dataset)) {
        for (const sample of dataset[label]) {
          const d = euclideanDistance(fv, sample);
          neighbors.push({ d, label });
        }
      }
      if (neighbors.length === 0) return null;
      neighbors.sort((a,b) => a.d - b.d);
      const k = Math.min(maxNeighbors, neighbors.length);
      const counts = {};
      for (let i = 0; i < k; i++) {
        const lab = neighbors[i].label;
        counts[lab] = (counts[lab] || 0) + 1;
      }
      let best = null, bestCount = -1;
      for (const l of Object.keys(counts)) {
        if (counts[l] > bestCount) { best = l; bestCount = counts[l]; }
      }
      return best;
    }

    function euclideanDistance(a, b) {
      let s = 0;
      for (let i = 0; i < a.length && i < b.length; i++) {
        const diff = a[i] - b[i];
        s += diff * diff;
      }
      return Math.sqrt(s);
    }

    let lastKeypoints = null; // último conjunto de keypoints para guardar/predicción

    // Iniciar cámara
    async function setupCamera() {
      try {
        statusEl.textContent = "Solicitando acceso a la cámara...";
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: 640, height: 480 },
        });
        video.srcObject = stream;
      } catch (err) {
        console.error('Error al acceder a la cámara:', err);
        statusEl.textContent = 'Error: No se pudo acceder a la cámara. Revisa permisos y que la página esté en HTTPS.';
        throw err;
      }
      await new Promise((resolve) => (video.onloadedmetadata = resolve));
      // Algunos navegadores requieren muted para autoplay (añadido en HTML)
      try { await video.play(); } catch (err) { console.warn('video.play() falló:', err); }

      // Ajustar tamaño real
      video.width = video.videoWidth;
      video.height = video.videoHeight;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    // Cargar modelo
    async function loadModel() {
      statusEl.textContent = 'Cargando modelo...';
      // Intentar usar backend WebGL para mejor rendimiento si está disponible
      try {
        if (tf && tf.setBackend) {
          await tf.setBackend('webgl');
          await tf.ready();
          console.log('TF backend:', tf.getBackend());
        }
      } catch (e) {
        console.warn('No se pudo cambiar backend TF (continuando):', e);
      }

      try {
        // Crear detector MediaPipe Hands (soporta maxHands)
        const modelType = handPoseDetection.SupportedModels.MediaPipeHands;
        const detectorConfig = {
          runtime: 'mediapipe',
          solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands',
          modelType: 'full',
          maxHands: 2,
        };
        detector = await handPoseDetection.createDetector(modelType, detectorConfig);
        console.log('✅ Detector cargado correctamente');
        statusEl.textContent = 'Modelo cargado. Coloca tus manos frente a la cámara.';
        detectHands();
      } catch (err) {
        console.error('Error cargando el modelo:', err);
        statusEl.textContent = 'Error: no se pudo cargar el modelo. Revisa la consola.';
        throw err;
      }
    }

    // Detección continua
    async function detectHands() {
      if (!detector) return;
      let predictions = [];
      try {
        // Detectar manos sin flipHorizontal para que coincidan exactamente con la pantalla
        predictions = await detector.estimateHands(video, { flipHorizontal: false });
      } catch (err) {
        console.error('Error en estimateHands:', err);
        statusEl.textContent = 'Error durante la detección. Revisa la consola.';
        requestAnimationFrame(detectHands);
        return;
      }

      // Dibujar imagen del video sin transformaciones
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      if (predictions.length > 0) {
        statusEl.textContent = `Manos detectadas: ${predictions.length}`;

        // conexiones sobre 21 puntos (MediaPipe)
        const connections = [
          [0, 1], [1, 2], [2, 3], [3, 4],
          [0, 5], [5, 6], [6, 7], [7, 8],
          [5, 9], [9, 10], [10, 11], [11, 12],
          [9, 13], [13, 14], [14, 15], [15, 16],
          [13, 17], [17, 18], [18, 19], [19, 20],
          [0, 17]
        ];

        // Dibujar cada mano
        lastKeypoints = null;
        for (let h = 0; h < predictions.length; h++) {
          const hand = predictions[h];
          const keypoints = hand.keypoints || [];

          // helper para obtener x,y en pixeles
          const getXY = (pt) => {
            let x = pt.x;
            let y = pt.y;
            // si vienen normalizados (0..1), convertir
            if (x <= 1 && y <= 1) {
              x = x * canvas.width;
              y = y * canvas.height;
            }
            return [x, y];
          };

          // dibujar conexiones
          ctx.strokeStyle = h === 0 ? 'cyan' : 'lime';
          ctx.lineWidth = 2;
          for (const [s, e] of connections) {
            if (!keypoints[s] || !keypoints[e]) continue;
            const [x1, y1] = getXY(keypoints[s]);
            const [x2, y2] = getXY(keypoints[e]);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }

          // dibujar puntos y números
          for (let i = 0; i < keypoints.length; i++) {
            const pt = keypoints[i];
            const [x, y] = getXY(pt);
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = h === 0 ? 'magenta' : 'orange';
            ctx.fill();
            ctx.fillStyle = 'yellow';
            ctx.font = '12px Arial';
            ctx.fillText(i.toString(), x + 6, y + 3);
          }
          // Usar la primera mano detectada para recolección/predicción
          if (h === 0) {
            lastKeypoints = keypoints.map(k => ({ x: k.x, y: k.y }));
            // Predecir con k-NN si hay dataset
            const fv = keypointsToFeature(lastKeypoints);
            const pred = knnPredict(fv);
            pushPrediction(pred);
          }
        }
      }

      // Si no detecta, mostrar mensaje tenue
      if (predictions.length === 0) {
        statusEl.textContent = 'No se detecta la mano. Asegúrate de buena iluminación y que la mano esté dentro del cuadro.';
        // empujar null para suavizado
        pushPrediction(null);
      }

      requestAnimationFrame(detectHands);
    }

    // Inicialización
    setupCamera()
      .then(() => loadModel())
      .catch((err) => console.error('Inicialización fallida:', err));
