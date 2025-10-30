from flask import Flask, request, render_template_string, redirect, url_for, send_from_directory, jsonify
from pathlib import Path
import tempfile
import os
import shutil
import traceback

# Import prediction helper from infer.py
from infer import build_transform, build_model, get_device, predict_one, pick_checkpoint

app = Flask(__name__)



ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "data" / "infer" / "web_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

HTML_PAGE = """
<!doctype html>
<html lang="en" data-bs-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Upload & Analyze – Proof Pixel</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    :root{
      --pp-bg:#0f1a2b;         /* deep navy */
      --pp-panel:#121f33;      /* panel navy */
      --pp-accent:#21d4a3;     /* mint accent */
      --pp-danger:#ff496a;     /* pink/red for DEEPFAKE */
      --pp-ok:#38e07d;         /* green for REAL */
      --pp-dash:#3a4863;
    }
    body{ background:var(--pp-bg); color:#e7edf5; }
    .topbar{
      background:rgba(255,255,255,0.02);
      border-bottom:1px solid rgba(255,255,255,0.06);
      backdrop-filter: blur(6px);
    }
    .brand{
      font-weight:800; letter-spacing:.5px;
    }
    .brand .pixel{ color:var(--pp-accent); }
    .hero{ padding:48px 0 12px; }
    .panel{
      background:var(--pp-panel); border:1px solid rgba(255,255,255,.06);
      border-radius:14px; box-shadow:0 10px 30px rgba(0,0,0,.25);
    }
    .dropzone{
      border:2px dashed var(--pp-dash);
      border-radius:12px; padding:38px; text-align:center;
      transition: all .2s ease; background:rgba(255,255,255,.02);
    }
    .dropzone.dragover{ border-color:var(--pp-accent); background:rgba(33,212,163,.08); }
    .dz-hint{ color:#99a8bf; }
    .preview{
      max-width:360px; width:100%; border-radius:10px; border:1px solid rgba(255,255,255,.08);
    }
    .result-line{
      font-size:1.15rem; font-weight:700; text-align:center; margin-top:16px;
    }
    .result-badge{ padding:.25rem .55rem; border-radius:.5rem; font-weight:800; }
    .badge-fake{ background:var(--pp-danger); color:#fff; }
    .badge-real{ background:var(--pp-ok); color:#0b1725; }
    .footer-note{ color:#8fa1bd; font-size:.9rem; }
    .btn-analyze{ background:var(--pp-accent); color:#0b1725; font-weight:700; }
    .metrics small{ color:#9bb0cb; }
  </style>
</head>
<body>
  <!-- topbar -->
  <nav class="navbar topbar">
    <div class="container-xxl">
      <span class="navbar-brand brand">
        PROOF<span class="pixel">PiXEL</span>
      </span>
      <div class="d-flex gap-2">
        <button class="btn btn-outline-light btn-sm d-none d-md-inline" disabled>Deepfake Detection</button>
        <button class="btn btn-success btn-sm">Logout</button>
      </div>
    </div>
  </nav>

  <div class="container-xxl">
    <section class="hero text-center">
      <h1 class="fw-bold">Analyze Image</h1>
      <p class="text-secondary mb-0">Upload an image file to check for manipulation. Our AI will provide a confidence score.</p>
    </section>

    <div class="row justify-content-center mt-3">
      <div class="col-lg-9">
        <div class="panel p-4 p-md-5">
          <!-- dropzone -->
          <div id="dropzone" class="dropzone mb-4">
            <input id="fileInput" type="file" accept="image/*" hidden>
            <p class="mb-1">Click here or drag & drop an image to upload</p>
            <small class="dz-hint">JPEG · PNG · WEBP</small>
          </div>

          <div class="d-flex gap-2 justify-content-center mb-3">
            <button id="analyzeBtn" class="btn btn-analyze px-4">Analyze Image</button>
            <button id="clearBtn" class="btn btn-outline-light px-4">Clear</button>
          </div>

          <div id="feedback" class="text-center"></div>

          <!-- result section -->
          <div id="resultWrap" class="mt-4" style="display:none;">
            <div class="row g-4 align-items-center justify-content-center">
              <div class="col-md-auto text-center">
                <img id="preview" class="preview" alt="preview">
              </div>
              <div class="col-md-6">
                <div class="metrics">
                  <div class="result-line">
                    Result: <span id="resultBadge" class="result-badge"></span>
                    <span class="ms-1">(Confidence: <span id="confPct">0%</span>)</span>
                  </div>
                  <hr class="border-secondary-subtle">
                  <div class="row gy-2">
                    <div class="col-6"><small>p_real</small><div id="pReal" class="fs-5 fw-bold">—</div></div>
                    <div class="col-6"><small>p_fake</small><div id="pFake" class="fs-5 fw-bold">—</div></div>
                    <div class="col-6"><small>ELA</small><div id="ela" class="fs-6">—</div></div>
                    <div class="col-6"><small>FFT ratio</small><div id="fft" class="fs-6">—</div></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <p class="footer-note mt-4 mb-0 text-center">Model runs on server. Check logs for device & checkpoint.</p>
        </div>
      </div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    const dz = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const feedback = document.getElementById('feedback');
    const resultWrap = document.getElementById('resultWrap');
    const preview = document.getElementById('preview');
    const resultBadge = document.getElementById('resultBadge');
    const confPct = document.getElementById('confPct');
    const pReal = document.getElementById('pReal');
    const pFake = document.getElementById('pFake');
    const ela = document.getElementById('ela');
    const fft = document.getElementById('fft');

    // click -> open file dialog
    dz.addEventListener('click', ()=> fileInput.click());

    // drag & drop UX
    ['dragenter','dragover'].forEach(evt =>
      dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.add('dragover'); })
    );
    ['dragleave','drop'].forEach(evt =>
      dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.remove('dragover'); })
    );
    dz.addEventListener('drop', e => {
      const files = e.dataTransfer.files;
      if (files && files[0]) { fileInput.files = files; showPreview(files[0]); }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files[0]) showPreview(fileInput.files[0]);
    });

    function showPreview(file){
      preview.src = URL.createObjectURL(file);
      resultWrap.style.display = 'none';
    }

    function resetUI(){
      fileInput.value = '';
      preview.removeAttribute('src');
      resultWrap.style.display = 'none';
      feedback.innerHTML = '';
      resultBadge.className = 'result-badge';
      confPct.textContent = '0%';
      pReal.textContent = '—';
      pFake.textContent = '—';
      ela.textContent = '—';
      fft.textContent = '—';
    }

    clearBtn.addEventListener('click', (e)=>{ e.preventDefault(); resetUI(); });

    analyzeBtn.addEventListener('click', async (e)=>{
      e.preventDefault();
      feedback.innerHTML = '';
      if (!fileInput.files || fileInput.files.length === 0){
        feedback.innerHTML = '<div class="alert alert-warning">Please select an image first.</div>';
        return;
      }
      const file = fileInput.files[0];
      const form = new FormData();
      form.append('file', file);

      analyzeBtn.disabled = true;
      analyzeBtn.innerText = 'Analyzing...';
      feedback.innerHTML = '<div class="spinner-border text-light spinner-border-sm me-2"></div> Running model...';

      try{
        const res = await fetch('/api/predict', { method:'POST', body: form });
        const data = await res.json();
        if(!res.ok){ throw new Error(data.error || res.statusText); }

        // numbers
        const conf = Number(data.confidence || 0);
        const preal = Number(data.p_real || 0);
        const pfake = Number(data.p_fake || 0);

        // badge
        const label = String(data.label || '').toUpperCase();
        resultBadge.textContent = label;
        if(label.includes('FAKE')){
          resultBadge.classList.add('badge-fake');
        } else {
          resultBadge.classList.add('badge-real');
        }
        confPct.textContent = (conf*100).toFixed(2) + '%';

        pReal.textContent = preal.toFixed(3);
        pFake.textContent = pfake.toFixed(3);
        ela.textContent   = Number(data.ela || 0).toFixed(3);
        fft.textContent   = Number(data.fft_ratio || 0).toFixed(3);

        resultWrap.style.display = 'block';
        feedback.innerHTML = '';
      }catch(err){
        feedback.innerHTML = '<div class="alert alert-danger">Error: '+ err.message +'</div>';
      }finally{
        analyzeBtn.disabled = false;
        analyzeBtn.innerText = 'Analyze Image';
      }
    });
  </script>
</body>
</html>
"""

def load_model(checkpoint: str = None, model_name: str = "resnet50"):
    device = get_device()
    ckpt = checkpoint or pick_checkpoint()
    tfm = build_transform(model_name)
    model = build_model(model_name, num_classes=2, ckpt_path=ckpt, device=device)
    return model, tfm, device


# Load model lazily on first request
_MODEL = None
_TFM = None
_DEVICE = None


@app.route('/', methods=['GET', 'POST'])
def upload():
    global _MODEL, _TFM, _DEVICE
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # save to uploads dir
        tmp_path = UPLOAD_DIR / file.filename
        file.save(tmp_path)

        try:
            if _MODEL is None:
                _MODEL, _TFM, _DEVICE = load_model()

            res = predict_one(_MODEL, _TFM, _DEVICE, tmp_path)
            # attach filename for display
            res['file_name'] = tmp_path.name
            result = type('R', (), res)
        except Exception as e:
            traceback.print_exc()
            result = type('R', (), {'label': 'error', 'confidence': 0.0, 'p_real': 0.0, 'p_fake': 0.0, 'ela':0.0, 'fft_ratio':0.0, 'file_name': tmp_path.name})

    return render_template_string(HTML_PAGE, result=result)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint: accepts a file under 'file' and returns JSON with prediction."""
    global _MODEL, _TFM, _DEVICE
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400

    tmp_dir = tempfile.mkdtemp(prefix='df_ui_')
    try:
        tmp_path = Path(tmp_dir) / file.filename
        file.save(tmp_path)
        if _MODEL is None:
            _MODEL, _TFM, _DEVICE = load_model()
        res = predict_one(_MODEL, _TFM, _DEVICE, tmp_path)
        # include filename so UI can preview — but don't expose full path
        res['file_name'] = file.filename
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


if __name__ == '__main__':
    # Run a development server
    app.run(host='0.0.0.0', port=5001, debug=True)
 