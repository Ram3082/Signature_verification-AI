from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os, cv2, numpy as np, pickle, json, sqlite3, base64
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "sigverify_secret_2024"

UPLOAD_FOLDER = "static/uploads"
DB_PATH       = "database.db"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/gradcam", exist_ok=True)

# ── Load metrics ──────────────────────────────────────────────────────────────
try:
    with open("metrics.json") as f:
        METRICS = json.load(f)
except FileNotFoundError:
    METRICS = {
        "accuracy":0,"precision":0,"recall":0,"f1":0,
        "sensitivity":0,"specificity":0,
        "cm":{"TN":0,"FP":0,"FN":0,"TP":0},
        "total_samples":0,"train_samples":0,"test_samples":0,
        "forged_count":0,"original_count":0,
        "best_model":"Random Forest","best_short":"RFC",
        "cnn_available":False,"all_models":{}
    }

# ── Try loading CNN (skip silently if out of memory) ──────────────────────────
CNN_AVAILABLE = False
cnn_model     = None

try:
    import tensorflow as tf
    # Only load if cnn_model.keras exists
    if os.path.exists("cnn_model.keras"):
        cnn_model     = tf.keras.models.load_model("cnn_model.keras")
        CNN_AVAILABLE = True
        print("✅ CNN model loaded")
    else:
        print("⚠️  cnn_model.keras not found — using classic model")
except Exception as e:
    print(f"⚠️  TensorFlow not available ({e}) — using classic model only")

# ── Load classic model ────────────────────────────────────────────────────────
classic_model = None
try:
    classic_model = pickle.load(open("model.pkl", "rb"))
    print("✅ Classic model loaded")
except Exception as e:
    print(f"❌ Classic model load failed: {e}")

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_used TEXT NOT NULL,
                gradcam_path TEXT DEFAULT '',
                timestamp TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)

init_db()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ═════════════════════════════════════════════════════════════════════════════
#  GRAD-CAM (only runs if CNN loaded successfully)
# ═════════════════════════════════════════════════════════════════════════════
def generate_gradcam(img_array, save_path):
    if not CNN_AVAILABLE or cnn_model is None:
        return None
    try:
        import tensorflow as tf

        last_conv = None
        for layer in reversed(cnn_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
        if last_conv is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs  = cnn_model.inputs,
            outputs = [cnn_model.get_layer(last_conv).output, cnn_model.output]
        )
        img_tensor = tf.cast(img_array[np.newaxis, ...], tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            pred_index  = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]

        grads        = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out     = conv_outputs[0]
        heatmap      = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap).numpy()
        heatmap      = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        orig_img        = (img_array * 255).astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap, (64, 64))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay         = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
        overlay_large   = cv2.resize(overlay, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_path, overlay_large)
        return save_path

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

# ═════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
def run_prediction(img_bgr, filename="upload"):
    img_resized  = cv2.resize(img_bgr, (64, 64))
    img_norm     = img_resized.astype("float32") / 255.0
    gradcam_path = None

    if CNN_AVAILABLE and cnn_model is not None:
        proba      = cnn_model.predict(img_norm[np.newaxis, ...], verbose=0)[0]
        pred       = int(np.argmax(proba))
        conf       = round(float(np.max(proba)) * 100, 1)
        model_name = "CNN"
        gc_fname   = f"gradcam_{os.path.splitext(filename)[0]}_{int(datetime.now().timestamp())}.jpg"
        gc_path    = os.path.join("static/gradcam", gc_fname)
        gradcam_path = generate_gradcam(img_norm, gc_path)

    elif classic_model is not None:
        flat       = img_resized.reshape(1, -1)
        pred       = int(classic_model.predict(flat)[0])
        proba      = classic_model.predict_proba(flat)[0]
        conf       = round(float(np.max(proba)) * 100, 1)
        model_name = METRICS.get("best_model", "RFC")
    else:
        return "Unknown", "forged", 0.0, None, "None"

    if pred == 1:
        return "Original Signature", "original", conf, gradcam_path, model_name
    return "Forged Signature", "forged", conf, gradcam_path, model_name

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def home():
    user = None
    if "user_id" in session:
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE id=?",
                                (session["user_id"],)).fetchone()
    return render_template("index.html", user=user)

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        name     = request.form.get("name","").strip()
        email    = request.form.get("email","").strip()
        password = request.form.get("password","")
        if not name or not email or not password:
            flash("All fields are required.", "error")
            return render_template("signup.html")
        try:
            with get_db() as conn:
                conn.execute("INSERT INTO users (name,email,password) VALUES (?,?,?)",
                             (name, email, generate_password_hash(password)))
            flash("Account created! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "error")
    return render_template("signup.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email","").strip()
        password = request.form.get("password","")
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email=?",
                                (email,)).fetchone()
        if user and check_password_hash(user["password"], password):
            session["user_id"]   = user["id"]
            session["user_name"] = user["name"]
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for("home"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/predict", methods=["POST"])
def predict():
    file     = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    img = cv2.imread(filepath)
    result, result_cls, confidence, gradcam_path, model_name = run_prediction(img, file.filename)
    if "user_id" in session:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO history (user_id,filename,result,confidence,model_used,gradcam_path) VALUES (?,?,?,?,?,?)",
                (session["user_id"], file.filename, result,
                 confidence, model_name, gradcam_path or "")
            )
    return render_template("result.html",
        prediction   = result,
        result_cls   = result_cls,
        confidence   = confidence,
        img_path     = filepath,
        gradcam_path = gradcam_path,
        model_name   = model_name,
        metrics      = METRICS,
        user_name    = session.get("user_name", "Guest"),
    )

@app.route("/history")
@login_required
def history():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM history WHERE user_id=? ORDER BY timestamp DESC",
            (session["user_id"],)
        ).fetchall()
    return render_template("history.html", history=rows,
                           user_name=session.get("user_name",""))

# ── REST API ──────────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if request.is_json:
            data     = request.get_json()
            nparr    = np.frombuffer(base64.b64decode(data.get("image","")), np.uint8)
            img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            filename = "api_upload.jpg"
        else:
            file = request.files.get("file")
            if not file:
                return jsonify({"error":"No file provided"}), 400
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            img = cv2.imread(filepath)

        if img is None:
            return jsonify({"error":"Could not read image"}), 400

        result, result_cls, confidence, _, model_name = run_prediction(img, filename)
        return jsonify({
            "success":    True,
            "prediction": result,
            "label":      result_cls,
            "confidence": confidence,
            "model":      model_name,
            "timestamp":  datetime.now().isoformat(),
            "filename":   filename,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    return jsonify(METRICS)

@app.route("/api/history", methods=["GET"])
def api_history():
    if "user_id" not in session:
        return jsonify({"error":"Not logged in"}), 401
    with get_db() as conn:
        rows = conn.execute(
            "SELECT filename,result,confidence,model_used,timestamp FROM history WHERE user_id=? ORDER BY timestamp DESC LIMIT 50",
            (session["user_id"],)
        ).fetchall()
    return jsonify({"history":[dict(r) for r in rows]})

if __name__ == "__main__":
    app.run(debug=True)