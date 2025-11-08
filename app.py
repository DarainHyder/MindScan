# app.py
# Streamlit MindCare with:
# - Top navigation bar (About | App)
# - About landing (aligned hero with circular photo + optional background + gallery)
# - App (Model Controls + Text Analysis + PHQ‑9 + Resources)
# Uses src/inference.py: DepressionDetector, PHQ9Handler

import os
import sys
import glob
import base64
from typing import List, Optional
from textwrap import dedent

import streamlit as st

# ----------- PROFILE / CONTENT CONFIG -----------
AUTHOR_NAME = "Syed Darain Hyder Kazmi"
AUTHOR_TITLE = "Data Scientist | ML Engineer"
AUTHOR_TAGLINE = "Building AI for mental health awareness and early screening."
AUTHOR_ABOUT = (
    "I'm passionate about applying machine learning and NLP to real human problems. "
    "This project explores how AI can support mental wellness by providing a private, "
    "educational screening tool that combines AI text analysis with the clinically validated PHQ‑9 questionnaire."
)
AUTHOR_INSPIRATION = (
    "The inspiration behind this project is to make early screening more accessible. "
    "While it is not a replacement for professional care, thoughtful technology can help people reflect, "
    "monitor, and take the first step towards getting help."
)
AUTHOR_TECH = [
    "Python • scikit‑learn • Transformers (DistilBERT)",
    "NLP preprocessing & explainability (feature contributions)",
    "Streamlit UI with dark health theme",
    "CI/CD ready (train baseline/transformer, artifact uploads)",
]
AUTHOR_LINKS = {
    "GitHub": "https://github.com/DarainHyder",
    "LinkedIn": "https://www.linkedin.com/in/syed-darain-hyder-kazmi/",
    "Email": "mailto:darainhyder21@gmail.com",
    "Portfolio": "https://darainhyder.netlify.app/",
}

# Image locations
AVATAR_CANDIDATES = [
    "assets/darain_1.jpg",
    "assets/darain_1.jpg",
    "assets/darain_1.jpg",
    "assets/darain_1.jpg",
]
HERO_BG_CANDIDATES = [
    "assets/hero_bg.jpg",
    "assets/hero_bg.png",
]
GALLERY_DIR = "assets/gallery"

# ----------- STREAMLIT SETUP -----------
st.set_page_config(
    page_title="MindCare - Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
)

# Import inference utilities
sys.path.append("src")
try:
    from inference import DepressionDetector, PHQ9Handler  # type: ignore
except Exception as e:
    DepressionDetector = None  # type: ignore
    PHQ9Handler = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

# ----------- DARK HEALTH THEME + TOP NAV CSS -----------
st.markdown(
    dedent("""
    <style>
    :root {
        --bg: #0b0f19;
        --bg-2: #0f1426;
        --panel: #12172b;
        --panel-2: #0f152a;
        --text: #e6e9ef;
        --muted: #9aa4b2;
        --accent: #7f5af0;   /* Neon purple */
        --accent-2: #2cb67d; /* Mint green */
        --warning: #ffd166;
        --danger: #ff4d6d;
        --chip: #1a2036;
        --border: #1a2341;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: radial-gradient(1200px circle at 10% 0%, var(--bg-2) 0%, var(--bg) 35%, #070b16 100%) !important;
        color: var(--text);
    }
    .block-container { padding-top: 0; padding-bottom: 2rem; }

    /* Top Navigation (now below header, sticky at 64px) */
    .topnav {
        position: sticky; top: 64px; z-index: 1000;
        height: 64px;
        background: rgba(10,14,25,0.65);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--border);
        border-radius: 12px;
    }
    .nav-inner {
        max-width: 1200px; margin: 0 auto;
        height: 64px; display: flex; align-items: center;
        justify-content: space-between; gap: 16px;
        padding: 0 16px;
    }
    .brand {
        display: flex; align-items: center; gap: 10px;
        font-weight: 900; letter-spacing: -0.3px;
        font-size: 1.2rem;
    }
    .brand .text {
        background: linear-gradient(90deg, var(--accent), #00d1b2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .links { display: flex; align-items: center; gap: 8px; }
    .nav-link {
        display: inline-flex; align-items: center; height: 38px;
        padding: 0 14px; border-radius: 999px; font-weight: 700;
        color: var(--muted); text-decoration: none; border: 1px solid transparent;
        transition: all .15s ease;
    }
    .nav-link:hover { color: var(--text); border-color: var(--border); background: rgba(255,255,255,0.03); }
    .nav-link.active {
        color: white;
        background: linear-gradient(135deg, var(--accent), #6a43e6);
        border: 1px solid #6a43e6;
        box-shadow: 0 10px 24px rgba(127,90,240,0.25);
    }
    .nav-cta {
        display: inline-flex; align-items: center; height: 38px;
        padding: 0 14px; border-radius: 999px; font-weight: 800;
        background: linear-gradient(135deg, var(--accent-2), #1fb37a);
        color: #071216; text-decoration: none; border: 1px solid #1fb37a;
        margin-left: 6px;
    }
    .nav-spacer { height: 12px; }

    /* Title */
    .title {
        font-size: clamp(2.2rem, 5vw, 3rem);
        font-weight: 800; letter-spacing: -0.5px; line-height: 1.1;
        margin: 2px 0 4px;
        background: linear-gradient(90deg, var(--accent), #00d1b2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        text-shadow: 0 6px 40px rgba(127, 90, 240, 0.25);
    }
    .subtitle { color: var(--muted); margin-bottom: 12px; }

    /* ECG divider */
    .ecg {
        position: relative; height: 22px; margin: 8px 0 18px;
        background: linear-gradient(90deg, rgba(127,90,240,0.15), rgba(44,182,125,0.15));
        border-radius: 999px; box-shadow: inset 0 0 0 1px rgba(26,35,65,0.6); overflow: hidden;
    }
    .ecg::before {
        content: ''; position: absolute; inset: 0;
        background: repeating-linear-gradient(90deg, transparent 0px, transparent 10px, rgba(255,255,255,0.04) 10px, rgba(255,255,255,0.04) 11px);
        animation: move 6s linear infinite; opacity: 0.6;
    }
    @keyframes move { 0% { transform: translateX(0);} 100% { transform: translateX(-60px);} }

    /* Inputs / buttons / metrics / progress */
    div.stButton > button {
        background: linear-gradient(135deg, var(--accent), #6a43e6);
        color: white; border: 1px solid #6a43e6;
        border-radius: 10px; padding: 0.6rem 1rem; font-weight: 700; transition: all .2s ease;
    }
    div.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 10px 30px rgba(127, 90, 240, 0.35); }
    div.stButton button[kind="secondary"] { background: linear-gradient(135deg, #253151, #1a2542); border: 1px solid var(--border); color: var(--text); }
    div[data-baseweb="textarea"] textarea, div[data-baseweb="select"] > div { background: var(--panel); color: var(--text); border-radius: 12px; border: 1px solid var(--border); }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, var(--accent), #00d1b2); }
    .stProgress > div { background-color: #1e2440; border-radius: 999px; overflow: hidden; border: 1px solid var(--border); }
    [data-testid="stMetricValue"] { color: var(--accent-2); }
    [data-testid="stMetricLabel"] { color: var(--muted); }

    /* Cards / hero */
    .card {
        background: linear-gradient(180deg, rgba(18,23,43,0.92) 0%, rgba(14,19,33,0.96) 100%);
        border: 1px solid var(--border); color: var(--text);
        padding: 18px 20px; border-radius: 16px; box-shadow: 0 8px 40px rgba(0,0,0,0.25); margin-bottom: 14px;
    }
    .hero {
        position: relative; border-radius: 20px; padding: 18px; margin: 10px 0 16px;
        background: linear-gradient(180deg, rgba(18,23,43,0.85) 0%, rgba(14,19,33,0.9) 100%);
        border: 1px solid var(--border); box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    }
    .hero-bg { position: absolute; inset: 0; border-radius: 20px; background-position: center; background-size: cover; opacity: 0.18; filter: saturate(1) contrast(1.05); }
    .hero-grid { position: relative; display: grid; grid-template-columns: 360px 1fr; gap: 18px; align-items: stretch; z-index: 1; }
    .hero-card { border-radius: 16px; background: linear-gradient(180deg, rgba(18,23,43,0.9) 0%, rgba(14,19,33,0.95) 100%); border: 1px solid var(--border); display: grid; place-items: center; min-height: 320px; box-shadow: 0 10px 40px rgba(0,0,0,0.25); }
    .hero-content { width: 100%; height: 100%; display: grid; align-content: center; padding: 22px; }
    .title-chip { display: inline-block; font-weight: 700; padding: 8px 14px; border-radius: 999px; background: linear-gradient(135deg, var(--accent), #6a43e6); color: white; border: 1px solid #6a43e6; margin-bottom: 12px; box-shadow: 0 10px 30px rgba(127,90,240,0.25); }
    .name { font-size: clamp(1.6rem, 3vw, 2.2rem); font-weight: 800; margin: 6px 0 8px; }
    .tagline { color: var(--muted); margin: 0 0 12px; }
    .links a { margin-right: 14px; color: #a78bfa; text-decoration: none; }
    .links a:hover { color: #c4b5fd; text-decoration: underline; }

    /* Avatar */
    .avatar-wrap { width: 220px; height: 220px; border-radius: 50%; background: conic-gradient(from 180deg, var(--accent), #00d1b2, var(--accent)); padding: 5px; display: grid; place-items: center; box-shadow: 0 12px 40px rgba(127,90,240,0.25); }
    .avatar { width: 100%; height: 100%; border-radius: 50%; object-fit: cover; border: 6px solid #0e1321; background: #0e1321; }
    .avatar-fallback { width: 100%; height: 100%; border-radius: 50%; display: grid; place-items: center; font-size: 3rem; font-weight: 800; color: var(--text); background: linear-gradient(135deg, #1a2542, #0f152a); border: 6px solid #0e1321; }

    /* Chips / results / gallery */
    .chip { display: inline-block; margin: 4px 6px 0 0; padding: 6px 12px; background: var(--chip); border-radius: 999px; border: 1px solid var(--border); font-weight: 600; font-size: 0.9rem; color: var(--text); box-shadow: inset 0 -1px 0 rgba(255,255,255,0.03); }
    .warn, .good, .crisis { background: linear-gradient(180deg, rgba(18,23,43,0.92) 0%, rgba(14,19,33,0.96) 100%); border: 1px solid var(--border); color: var(--text); padding: 18px 20px; border-radius: 16px; box-shadow: 0 8px 40px rgba(0,0,0,0.25); margin: 6px 0 10px; }
    .warn { border-left: 6px solid var(--warning); } .good { border-left: 6px solid var(--accent-2); } .crisis { border-left: 6px solid var(--danger); }
    .gallery { display: grid; grid-auto-flow: column; gap: 14px; overflow-x: auto; padding-bottom: 6px; scrollbar-width: thin; }
    .gitem { width: 360px; height: 200px; border-radius: 16px; overflow: hidden; border: 1px solid var(--border); box-shadow: 0 10px 30px rgba(0,0,0,0.25); background: #0f152a; }
    .gitem img { width: 100%; height: 100%; object-fit: cover; display: block; }

    .small-muted { color: var(--muted); font-size: 0.9rem; }

    /* Tabs */
    [role="tablist"] { border-bottom: 1px solid var(--border); margin-bottom: 10px; }
    [role="tab"] { color: var(--muted); }
    [role="tab"][aria-selected="true"] { color: var(--text); border-bottom: 3px solid var(--accent); }

    /* Responsive hero */
    @media (max-width: 1000px) {
      .hero-grid { grid-template-columns: 1fr; }
      .hero-card { min-height: 280px; }
    }
    </style>
    """),
    unsafe_allow_html=True,
)

# ----------- SESSION STATE -----------
if "detector" not in st.session_state:
    st.session_state.detector = None
if "model_type" not in st.session_state:
    st.session_state.model_type = "baseline"
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5

# page: use query params for top nav
def _get_page() -> str:
    try:
        params = st.query_params
    except Exception:
        params = {}
    page = params.get("page", "about").lower()
    if page not in ("about", "app"):
        page = "about"
    # keep URL clean/consistent
    try:
        st.query_params["page"] = page
    except Exception:
        pass
    return page

def _set_page(page: str):
    try:
        st.query_params["page"] = page
    except Exception:
        pass

# ----------- HELPERS -----------
def html(s: str):
    st.markdown(dedent(s).strip(), unsafe_allow_html=True)

def _file_to_b64(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def _find_first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

def avatar_html(name: str) -> str:
    path = _find_first(AVATAR_CANDIDATES)
    if path:
        b64 = _file_to_b64(path)
        if b64:
            ext = os.path.splitext(path)[1].lstrip(".").lower()
            return f'<div class="avatar-wrap"><img class="avatar" src="data:image/{ext};base64,{b64}" alt="avatar"/></div>'
    initials = "".join([s[0].upper() for s in name.split()[:2]]) or "U"
    return f'<div class="avatar-wrap"><div class="avatar-fallback">{initials}</div></div>'

def hero_background_css() -> str:
    path = _find_first(HERO_BG_CANDIDATES)
    if not path:
        return ""
    b64 = _file_to_b64(path)
    if not b64:
        return ""
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    return f'<div class="hero-bg" style="background-image: url(data:image/{ext};base64,{b64});"></div>'

def load_gallery_images() -> List[str]:
    if not os.path.isdir(GALLERY_DIR):
        return []
    imgs = []
    for p in sorted(glob.glob(os.path.join(GALLERY_DIR, "*"))):
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            b64 = _file_to_b64(p)
            if b64:
                ext = os.path.splitext(p)[1].lstrip(".").lower()
                imgs.append(f"data:image/{ext};base64,{b64}")
    return imgs

def load_model(model_type: str) -> Optional["DepressionDetector"]:
    if _IMPORT_ERROR is not None:
        st.error(f"Could not import from src/: {_IMPORT_ERROR}")
        return None
    if st.session_state.detector is None or st.session_state.model_type != model_type:
        with st.spinner(f"Loading {model_type} model..."):
            try:
                detector = DepressionDetector(model_type=model_type)
                if hasattr(detector, "threshold"):
                    detector.threshold = float(st.session_state.threshold)
                st.session_state.detector = detector
                st.session_state.model_type = model_type
                st.success(f"Model initialized: {model_type}")
            except Exception as e:
                st.session_state.detector = None
                st.error(f"Error loading model: {e}. Ensure models/ contains trained artifacts.")
                return None
    if st.session_state.detector is not None and hasattr(st.session_state.detector, "threshold"):
        st.session_state.detector.threshold = float(st.session_state.threshold)
    return st.session_state.detector

def crisis_box():
    html("""
    <div class="crisis">
      <strong>🚨 Immediate Support</strong><br>
      If you're experiencing a mental health crisis or having thoughts of self-harm, please reach out immediately:<br><br>
      • US: Call or text <strong>988</strong><br>
      • Crisis Text Line: Text <strong>HOME</strong> to <strong>741741</strong><br>
      • International: <a href="https://findahelpline.com" target="_blank">findahelpline.com</a>
    </div>
    """)

def disclaimer_box():
    html("""
    <div class="warn">
      <strong>⚠️ Important Medical Disclaimer</strong><br>
      This platform provides educational screening tools only and is NOT a substitute for professional medical diagnosis or treatment.
    </div>
    """)

# ----------- HEADER -----------
html("<div class='title'>MindCare</div>")
html("<div class='subtitle'>AI‑powered mental health screening — Text Analysis + PHQ‑9</div>")
html('<div class="ecg"></div>')

# ----------- NAV (now below header) -----------
current_page = _get_page()
nav_html = f"""
<div class="topnav">
  <div class="nav-inner">
    <div class="brand">
      <span>🧠</span>
      <span class="text">MindCare</span>
    </div>
    <div class="links">
      <a class="nav-link {'active' if current_page=='about' else ''}" href="?page=about">About</a>
      <a class="nav-link {'active' if current_page=='app' else ''}" href="?page=app">App</a>
      <a class="nav-cta" href="https://github.com/your-handle" target="_blank">GitHub</a>
    </div>
  </div>
</div>
<div class="nav-spacer"></div>
"""
html(nav_html)

# ----------- PAGES -----------
if current_page == "about":
    # Hero section
    links_html = "".join([f'<a href="{v}" target="_blank">🔗 {k}</a>' for k, v in AUTHOR_LINKS.items()])
    hero_html = f"""
    <div class="hero">
      {hero_background_css()}
      <div class="hero-grid">
        <div class="hero-card">
          {avatar_html(AUTHOR_NAME)}
        </div>
        <div class="hero-card">
          <div class="hero-content">
            <div class="title-chip">{AUTHOR_TITLE}</div>
            <div class="name">{AUTHOR_NAME}</div>
            <div class="tagline">{AUTHOR_TAGLINE}</div>
            <div class="links">{links_html}</div>
          </div>
        </div>
      </div>
    </div>
    """
    html(hero_html)

    col_ai, col_insp = st.columns(2)
    with col_ai:
        html("<div class='card'>")
        st.subheader("About Me")
        st.write(AUTHOR_ABOUT)
        html("</div>")

    with col_insp:
        html("<div class='card'>")
        st.subheader("Inspiration")
        st.write(AUTHOR_INSPIRATION)
        html("</div>")

    html("<div class='card'>")
    st.subheader("Tech Stack")
    st.write("- " + "\n- ".join(AUTHOR_TECH))
    html("</div>")

    gallery_imgs = load_gallery_images()
    if gallery_imgs:
        st.subheader("Project Gallery")
        html("<div class='gallery'>")
        for src in gallery_imgs:
            html(f"<div class='gitem'><img src='{src}'/></div>")
        html("</div>")
        html("<span class='small-muted'>Add images to assets/gallery/</span>")

    c3, c4, c5 = st.columns([2, 2, 1])
    with c3:
        if st.button("Start MindCare →", use_container_width=True):
            _set_page("app")
            st.experimental_rerun()

else:
    # App page
    # Model controls at top
    with st.expander("Model Controls", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.session_state.model_type = st.selectbox(
                "Model type",
                options=["baseline", "transformer"],
                index=0 if st.session_state.model_type == "baseline" else 1,
                help="Baseline: TF‑IDF + Logistic Regression, Transformer: DistilBERT",
            )
        with col2:
            st.session_state.threshold = st.slider(
                "Decision threshold",
                min_value=0.1, max_value=0.9, value=float(st.session_state.threshold), step=0.05,
                help="Lower = higher recall, higher = higher precision.",
            )

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Load / Reload", use_container_width=True):
                load_model(st.session_state.model_type)
        with c2:
            if st.button("Unload", type="secondary", use_container_width=True):
                st.session_state.detector = None
                st.info("Model unloaded.")
        with c3:
            st.write(f"Model loaded: {'✅' if st.session_state.detector is not None else '❌'}")

    disclaimer_box()

    tab_overview, tab_text, tab_phq9, tab_resources = st.tabs(
        ["Overview", "Text Analysis", "PHQ‑9", "Resources"]
    )

    with tab_overview:
        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("Assessment Tools", "2")
        with cB:
            st.metric("Confidential", "100%")
        with cC:
            st.metric("Availability", "24/7")

        st.write(
            "MindCare offers two complementary approaches: free‑form AI text analysis and the clinically validated PHQ‑9 questionnaire. "
            "Use the Model Controls to load a model and adjust the decision threshold."
        )

    with tab_text:
        st.subheader("AI Text Analysis")
        user_text = st.text_area("Your text", placeholder="I've been feeling...", height=160)

        colA, colB = st.columns([1, 2])
        with colA:
            analyze = st.button("🔍 Analyze my text", use_container_width=True)
        with colB:
            auto_load = st.checkbox(
                "Auto-load model if not loaded",
                value=True,
                help="If enabled, clicking Analyze will try to load the model automatically.",
            )

        if analyze:
            if st.session_state.detector is None:
                detector = load_model(st.session_state.model_type) if auto_load else None
            else:
                detector = st.session_state.detector

            if detector is None:
                st.error("No model is loaded. Please open Model Controls and load a model.")
            elif not user_text or len(user_text.strip()) < 10:
                st.warning("Please enter at least 10 characters.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        result = detector.predict(user_text.strip(), return_explanation=True)
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        result = None

                if result:
                    risk = str(result.get("risk_level", "low"))
                    prob = float(result.get("probability", 0.0))
                    label = int(result.get("label", 0))

                    if risk in ("high", "very_high"):
                        crisis_box()

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Depression Indicator", "Positive" if label == 1 else "Negative")
                    with c2:
                        st.metric("Confidence", f"{prob * 100:.1f}%")
                    with c3:
                        st.metric("Risk Level", risk.replace("_", " ").title())

                    st.progress(min(100, max(0, int(prob * 100))))

                    explanation = result.get("explanation", [])
                    if explanation:
                        st.markdown("#### Key indicators")
                        chips_html = "".join(f'<span class="chip">{e.get("word","")}</span>' for e in explanation[:10])
                        html(chips_html)

                    st.markdown("#### Next steps")
                    if prob >= 0.6:
                        st.write(
                            "- Consider taking the PHQ‑9 questionnaire below\n"
                            "- Speak with a licensed mental health professional\n"
                            "- Reach out to trusted friends or family for support"
                        )
                    else:
                        st.write(
                            "- Continue monitoring your mental wellness\n"
                            "- Practice self-care and stress management\n"
                            "- Seek help if your situation changes"
                        )

    with tab_phq9:
        st.subheader("PHQ‑9 Depression Screening")
        PHQ9_QUESTIONS: List[str] = [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
            "Trouble concentrating on things, such as reading the newspaper or watching television",
            "Moving or speaking so slowly that other people could have noticed; or the opposite — being so fidgety or restless",
            "Thoughts that you would be better off dead, or of hurting yourself",
        ]

        options = [
            ("Select...", None),
            ("0 — Not at all", 0),
            ("1 — Several days", 1),
            ("2 — More than half the days", 2),
            ("3 — Nearly every day", 3),
        ]

        with st.form("phq9_form", clear_on_submit=False):
            responses: List[Optional[int]] = []
            for i, q in enumerate(PHQ9_QUESTIONS):
                sel = st.selectbox(
                    f"{i+1}. {q}",
                    options=options,
                    index=0,
                    format_func=lambda x: x[0] if isinstance(x, tuple) else str(x),
                    key=f"phq9_q_{i}",
                )
                responses.append(sel[1] if isinstance(sel, tuple) else None)
            submitted = st.form_submit_button("📊 Calculate score")

        if submitted:
            if PHQ9Handler is None:
                st.error("Could not import PHQ9Handler. Ensure src/inference.py is present.")
            elif any(r is None for r in responses):
                st.warning("Please answer all 9 questions before calculating your score.")
            else:
                try:
                    result = PHQ9Handler.score_phq9(responses)  # type: ignore
                except Exception as e:
                    st.error(f"Error scoring PHQ‑9: {e}")
                    result = None

                if result:
                    total = int(result.get("total_score", 0))
                    severity = str(result.get("severity", "unknown")).replace("_", " ").title()
                    emerg = bool(result.get("emergency_flag", False))

                    box_class = "good"
                    if total >= 15:
                        box_class = "crisis"
                    elif total >= 10:
                        box_class = "warn"

                    html(f"""
                    <div class="{box_class}">
                      <strong>PHQ‑9 Results</strong><br>
                      Total Score: <strong>{total} / 27</strong><br>
                      Severity: <strong>{severity}</strong>
                    </div>
                    """)

                    if emerg:
                        crisis_box()
                        st.warning("Question 9 flagged. Please seek immediate help if you're in danger.")

                    st.markdown("#### Interpretation")
                    if total <= 4:
                        st.write("Minimal or no depression.")
                    elif total <= 9:
                        st.write("Mild depression.")
                    elif total <= 14:
                        st.write("Moderate depression.")
                    elif total <= 19:
                        st.write("Moderately severe depression.")
                    else:
                        st.write("Severe depression.")

                    st.markdown("#### Recommendations")
                    st.write(
                        "- Consider sharing these results with a healthcare provider\n"
                        "- Explore the Resources tab for support options\n"
                        "- Remember: seeking help is a sign of strength"
                    )

    with tab_resources:
        st.subheader("Mental Health Resources")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Crisis Support")
            st.write(
                "- US: Call or text 988\n"
                "- Crisis Text Line: Text HOME to 741741\n"
                "- International: https://findahelpline.com"
            )
            st.markdown("##### Professional Help")
            st.write(
                "- Psychology Today: Find therapists near you\n"
                "- BetterHelp: Online therapy platform\n"
                "- SAMHSA: 1-800-662-4357 (treatment referral)"
            )
        with col2:
            st.markdown("##### Educational Resources")
            st.write(
                "- NAMI: Mental health education\n"
                "- Mental Health America: Screening tools\n"
                "- NIMH: Research and information"
            )
            st.markdown("##### Support Groups")
            st.write(
                "- DBSA: Depression support groups\n"
                "- ADAA: Anxiety support groups\n"
                "- 7 Cups: Free online counseling"
            )

html("<p class='small-muted'>This tool is for educational and research purposes only and is not a substitute for professional medical advice.</p>")