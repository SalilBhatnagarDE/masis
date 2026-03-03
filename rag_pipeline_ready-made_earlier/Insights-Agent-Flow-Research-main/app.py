"""
AI Leadership Insight Agent — Streamlit UI

Features:
- Chat interface with conversation history
- Document upload (PDF, DOCX, PPTX, TXT, MD, Images)
- One-click ingestion sidebar
- Real-time thought trace showing pipeline nodes
- Interactive Plotly charts when answer contains numerical data
- Concise report-style answers with document references
- Sample questions as quick-start buttons
"""

import os
import sys
import streamlit as st
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
logging.basicConfig(level=logging.INFO)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Leadership Insight Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem; font-weight: 800;
        margin-bottom: 0; letter-spacing: -0.5px;
    }
    .subtitle { color: #8892a4; font-size: 1rem; margin-top: -8px; margin-bottom: 24px; }
    .thought-trace {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px; padding: 14px 18px; margin: 8px 0 16px 0;
        border-left: 4px solid #667eea;
        font-family: 'Fira Code', 'Consolas', monospace;
        font-size: 0.82rem; color: #e0e0e0;
    }
    .thought-trace b { color: #a5b4fc; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #f8f9fc 0%, #eef1f6 100%); }
    .stButton > button {
        width: 100%; text-align: left; border-radius: 10px;
        border: 1px solid #e0e4f0; background: white;
        padding: 10px 16px; transition: all 0.2s ease; font-size: 0.88rem;
    }
    .stButton > button:hover {
        border-color: #667eea; background: #f5f7ff;
        transform: translateY(-1px); box-shadow: 0 2px 8px rgba(102,126,234,0.15);
    }
</style>
""", unsafe_allow_html=True)


# ─── Interactive Plotly Charts ────────────────────────────────────────────────

def render_chart(chart_data: dict):
    """Render an interactive Plotly chart from extracted chart data."""
    try:
        import plotly.graph_objects as go

        chart_type = chart_data.get("chart_type", "bar")
        labels = chart_data.get("labels", [])
        values = chart_data.get("values", [])
        title = chart_data.get("title", "")
        unit = chart_data.get("unit", "")

        if not labels or not values:
            return

        # Premium color palette with gradients
        colors = [
            '#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe',
            '#00f2fe', '#43e97b', '#fa709a', '#fee140', '#30cfd0',
        ]

        fig = go.Figure()

        if chart_type == "bar":
            fig.add_trace(go.Bar(
                x=labels, y=values,
                marker=dict(
                    color=colors[:len(labels)],
                    line=dict(width=0),
                    cornerradius=6,
                ),
                text=[f"{v:,.1f}" for v in values],
                textposition='outside',
                textfont=dict(size=13, color='white', family='Inter'),
                hovertemplate="<b>%{x}</b><br>%{y:,.1f} " + unit + "<extra></extra>",
            ))

        elif chart_type == "pie":
            fig.add_trace(go.Pie(
                labels=labels, values=values,
                marker=dict(colors=colors[:len(labels)], line=dict(color='#0e1117', width=2)),
                textinfo='label+percent',
                textfont=dict(size=12, color='white', family='Inter'),
                hole=0.4,  # Donut chart for modern look
                hovertemplate="<b>%{label}</b><br>%{value:,.1f} " + unit + " (%{percent})<extra></extra>",
            ))

        elif chart_type == "line":
            fig.add_trace(go.Scatter(
                x=labels, y=values,
                mode='lines+markers+text',
                line=dict(color='#667eea', width=3, shape='spline'),
                marker=dict(size=10, color='#764ba2', line=dict(color='white', width=2)),
                text=[f"{v:,.1f}" for v in values],
                textposition='top center',
                textfont=dict(size=11, color='#a5b4fc', family='Inter'),
                fill='tozeroy',
                fillcolor='rgba(102,126,234,0.1)',
                hovertemplate="<b>%{x}</b><br>%{y:,.1f} " + unit + "<extra></extra>",
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='white', family='Inter'), x=0.5),
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            font=dict(color='#8892a4', family='Inter'),
            margin=dict(l=30, r=30, t=50, b=30),
            height=350,
            width=560,
            xaxis=dict(
                showgrid=False, color='#8892a4',
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                showgrid=True, gridcolor='rgba(136,146,164,0.12)',
                color='#8892a4', title=unit if chart_type != "pie" else "",
                tickfont=dict(size=11),
            ),
            hoverlabel=dict(
                bgcolor='#1a1a2e', bordercolor='#667eea',
                font=dict(color='white', size=13, family='Inter'),
            ),
            showlegend=False,
        )

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.plotly_chart(fig, width='content', config={
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'displaylogo': False,
            })

    except ImportError:
        st.info("📊 Chart data available but plotly not installed. Run: `pip install plotly`")
    except Exception as e:
        st.warning(f"⚠️ Chart rendering error: {e}")


# ─── Initialize Session State ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "charts" not in st.session_state:
    st.session_state.charts = {}
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "ingest_stats" not in st.session_state:
    st.session_state.ingest_stats = {}
if "active_company" not in st.session_state:
    st.session_state.active_company = config.DEFAULT_COMPANY


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏢 Company")
    company_options = list(config.AVAILABLE_COMPANIES.keys())
    selected_company = st.selectbox(
        "Select company",
        options=company_options,
        index=company_options.index(st.session_state.active_company),
        format_func=lambda x: x.title(),
    )
    if selected_company != st.session_state.active_company:
        st.session_state.active_company = selected_company
        st.session_state.messages = []
        st.session_state.charts = {}
        st.session_state.ingested = False
        st.session_state.ingest_stats = {}
        config.set_active_company(selected_company)
        st.rerun()
    else:
        config.set_active_company(selected_company)

    st.divider()
    st.markdown("### 📁 Document Management")

    uploaded_files = st.file_uploader(
        "Upload company documents",
        type=["pdf", "docx", "pptx", "txt", "md",
              "jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        help="Upload PDFs, Word docs, presentations, text, Markdown, or images",
    )

    if uploaded_files:
        doc_dir = config.COMPANY_DOCS_DIR
        os.makedirs(doc_dir, exist_ok=True)
        saved = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(doc_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            saved.append(uploaded_file.name)
        st.success(f"✅ Uploaded {len(saved)} file(s) to {selected_company.title()}")

    doc_dir = config.COMPANY_DOCS_DIR
    if os.path.exists(doc_dir):
        existing = [f for f in os.listdir(doc_dir) if not f.startswith('.')]
        if existing:
            with st.expander(f"📄 {selected_company.title()} Documents ({len(existing)})"):
                for fname in existing:
                    fsize = os.path.getsize(os.path.join(doc_dir, fname))
                    size_str = f"{fsize/1024:.0f} KB" if fsize < 1048576 else f"{fsize/1048576:.1f} MB"
                    st.markdown(f"- **{fname}** ({size_str})")

    st.divider()

    force_reingest = st.checkbox("Force re-ingest", value=False,
                                  help="Bypass cache and re-process all documents")
    if st.button("🔄 Ingest Documents", type="primary", width='stretch'):
        with st.spinner(f"⏳ Processing {selected_company.title()} documents..."):
            try:
                from ingestion.ingest import ingest_pipeline
                result = ingest_pipeline(force=force_reingest)
                if result.get("skipped"):
                    st.info(
                        f"⏭️ No document changes detected — ingestion skipped.\n\n"
                        f"Check **Force re-ingest** to re-process all files."
                    )
                else:
                    st.session_state.ingested = True
                    st.session_state.ingest_stats = result
                    st.success(
                        f"✅ Ingestion Complete!\n\n"
                        f"- **Pages:** {result.get('num_documents', 0)}\n"
                        f"- **Chunks:** {result.get('num_chunks', 0)}"
                    )
            except Exception as e:
                st.error(f"❌ Ingestion Error: {e}")

    st.divider()
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")):
        st.markdown(f"🟢 **Status:** {selected_company.title()} documents indexed")
        st.session_state.ingested = True
    else:
        st.markdown("🟡 **Status:** No documents indexed")

    st.divider()

    with st.expander("🏗️ Pipeline Architecture"):
        st.markdown("""
        1. 🔀 **Router** — Intent classification
        2. ✏️ **Rewrite** — HyDE query optimization
        3. 🏷️ **Metadata** — Extract filters
        4. 🔍 **Retrieve** — Vector + BM25 + RRF
        5. ⚡ **Rerank** — Cross-encoder scoring
        6. ✅ **Grade** — CRAG relevance check
        7. 📝 **Generate** — Concise factual answer
        8. 🔬 **Verify** — Self-RAG hallucination check
        9. 📊 **Enrich** — Charts + Report + References
        """)

    st.divider()
    if st.button("🗑️ Clear Chat", width='stretch'):
        st.session_state.messages = []
        st.session_state.charts = {}
        st.rerun()


# ─── Main Content ────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🧠 AI Leadership Insight Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Hard facts from your documents — with charts, references, and source tracking</p>',
    unsafe_allow_html=True,
)

# ─── Sample Questions (click to run) ─────────────────────────────────────────
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if not st.session_state.messages:
    st.markdown("### 💡 Quick Start — click to run:")
    samples = [
        "What is our current revenue trend?",
        "Which departments are underperforming?",
        "What were the key risks highlighted in the last year?",
    ]
    for i, q in enumerate(samples):
        if st.button(f"▶ {q}", key=f"sample_{i}", width='stretch'):
            st.session_state.pending_question = q
            st.rerun()

# ─── Chat Messages ───────────────────────────────────────────────────────────
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and i in st.session_state.charts:
            render_chart(st.session_state.charts[i])

# ─── Chat Input ──────────────────────────────────────────────────────────────
# Handle both typed input and sample button clicks
prompt = st.chat_input("Ask about your company documents...")

# If a sample question was clicked, use that instead
if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thought_container = st.empty()
        answer_container = st.empty()
        chart_container = st.empty()

        try:
            from graph.workflow import stream_agent

            thoughts = []
            final_answer = ""
            final_chart = {}

            node_icons = {
                "router": "🔀 Router",
                "rewrite_query": "✏️ Rewriting",
                "extract_metadata": "🏷️ Metadata",
                "retrieve": "🔍 Search",
                "rerank": "⚡ Rerank",
                "grade_documents": "✅ Grade",
                "generate": "📝 Generate",
                "check_hallucination": "🔬 Verify",
                "enrich_answer": "📊 Enrich",
                "clarify": "❓ Clarify",
                "fallback": "⚠️ Fallback",
            }

            for node_name, state_update in stream_agent(prompt):
                icon = node_icons.get(node_name, f"⚙️ {node_name}")
                thoughts.append(icon)

                trace_text = " → ".join(thoughts)
                thought_container.markdown(
                    f'<div class="thought-trace">🧠 <b>Pipeline:</b> {trace_text}</div>',
                    unsafe_allow_html=True,
                )

                if "generation" in state_update and state_update["generation"]:
                    final_answer = state_update["generation"]

                if "chart_data" in state_update and state_update["chart_data"]:
                    final_chart = state_update["chart_data"]

            if final_answer:
                answer_container.markdown(final_answer)

                if final_chart:
                    with chart_container.container():
                        render_chart(final_chart)

                msg_index = len(st.session_state.messages)
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer}
                )
                if final_chart:
                    st.session_state.charts[msg_index] = final_chart
            else:
                answer_container.warning("No answer was generated. Please try again.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logging.exception("Error in agent pipeline")
