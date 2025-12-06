import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import faiss

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="🧠 DocuMind AI", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 24px;
        padding: 60px 40px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.95);
        font-weight: 400;
        margin-bottom: 20px;
    }
    
    .hero-description {
        font-size: 1.05rem;
        color: rgba(255, 255, 255, 0.85);
        max-width: 900px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 10px;
    }
    
    .feature-description {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.7);
        line-height: 1.5;
    }
    
    /* Upload Section */
    .upload-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 24px;
        padding: 60px 40px;
        text-align: center;
        margin: 40px 0;
        box-shadow: 0 20px 60px rgba(245, 87, 108, 0.3);
    }
    
    .upload-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 15px;
    }
    
    .upload-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 10px;
    }
    
    .upload-hint {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.75);
        margin-top: 10px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }
    
    /* File Uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        padding: 20px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        padding: 12px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Text Input */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        font-size: 1rem;
        padding: 12px 16px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.3);
    }
    
    /* Select Box */
    .stSelectbox>div>div {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
    }
    
    /* Number Input */
    .stNumberInput>div>div>input {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        color: white;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.12);
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        border-left: 4px solid #667eea;
        color: white;
    }
    
    /* Stats Cards */
    .stats-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 5px;
    }
    
    .stats-label {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Markdown text */
    .element-container p, .element-container li {
        color: rgba(255, 255, 255, 0.85);
    }
    
    h1, h2, h3 {
        color: white !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    """Load and cache AI models"""
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline

        with st.spinner("🔄 Loading AI models..."):
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=-1
            )
        return embedding_model, summarizer, None
    except Exception as e:
        return None, None, str(e)

embedding_model, summarizer, error = load_models()

if error:
    st.error(f"❌ Error loading models: {error}")
    st.info("Please install: `pip install sentence-transformers transformers torch pymupdf pillow faiss-cpu`")
    st.stop()

# -------------------- SESSION STATE --------------------
def init_session_state():
    """Initialize all session state variables"""
    if 'doc_chunks' not in st.session_state:
        st.session_state.doc_chunks = []
    if 'doc_embeddings' not in st.session_state:
        st.session_state.doc_embeddings = []
    if 'metadata' not in st.session_state:
        st.session_state.metadata = []
    if 'pdf_store' not in st.session_state:
        st.session_state.pdf_store = {}
    if 'pdf_summaries' not in st.session_state:
        st.session_state.pdf_summaries = {}
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
    if 'last_uploaded_files' not in st.session_state:
        st.session_state.last_uploaded_files = []
    if 'app_started' not in st.session_state:
        st.session_state.app_started = False

init_session_state()

# -------------------- BACKEND FUNCTIONS (UNCHANGED) --------------------
def chunk_text(text, chunk_size=400, overlap=50):
    """Split text into overlapping chunks for better context"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def generate_summary(text, max_length=150):
    """Generate summary using the summarization model"""
    try:
        text = ' '.join(text.split())
        word_count = len(text.split())
        
        if word_count < 40:
            return f"📄 Document contains {word_count} words. Too short for automatic summarization."
        
        max_words = 800
        if word_count > max_words:
            text = ' '.join(text.split()[:max_words])
            word_count = max_words
        
        adjusted_max_length = min(max_length, int(word_count * 0.35))
        adjusted_min_length = min(30, int(adjusted_max_length * 0.5))
        
        summary = summarizer(
            text,
            max_length=adjusted_max_length,
            min_length=adjusted_min_length,
            do_sample=False,
            truncation=True
        )
        return summary[0]['summary_text']
    except Exception as e:
        sentences = text.split('.')[:3]
        return '. '.join(sentences).strip() + '.'

def extract_text_from_pdf(file, filename):
    """Extract text, create chunks, embeddings, and generate summary"""
    try:
        file_bytes = file.read()
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        st.session_state.pdf_store[filename] = pdf
        
        full_text = ""
        page_count = 0
        
        for i, page in enumerate(pdf):
            text = page.get_text()
            if text.strip():
                full_text += text + " "
                page_count += 1
                
                chunks = chunk_text(text, chunk_size=400, overlap=50)
                for chunk in chunks:
                    emb = embedding_model.encode(chunk, convert_to_tensor=False)
                    st.session_state.doc_chunks.append(chunk)
                    st.session_state.doc_embeddings.append(emb)
                    st.session_state.metadata.append({
                        "filename": filename,
                        "page": i + 1,
                        "chunk": chunk
                    })
        
        if full_text.strip():
            summary = generate_summary(full_text, max_length=150)
            stats = f"\n\n📊 **Stats:** {page_count} pages | {len(full_text.split())} words"
            summary = summary + stats
        else:
            summary = "⚠️ No text content found in document."
        
        st.session_state.pdf_summaries[filename] = summary
        return summary, page_count
    except Exception as e:
        error_msg = f"❌ Error processing PDF: {str(e)}"
        st.session_state.pdf_summaries[filename] = error_msg
        return error_msg, 0

def build_faiss_index():
    """Build FAISS index for semantic search"""
    try:
        if st.session_state.doc_embeddings:
            embeddings_array = np.array(st.session_state.doc_embeddings).astype('float32')
            dimension = embeddings_array.shape[1]
            
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings_array)
            index.add(embeddings_array)
            
            st.session_state.faiss_index = index
            return True
    except Exception as e:
        st.error(f"❌ Error building FAISS index: {e}")
        return False

def semantic_search(query, k=5, filter_filename=None, filter_page=None):
    """Perform semantic search with optional filters"""
    if not st.session_state.faiss_index or not st.session_state.doc_embeddings:
        return []
    
    try:
        query_vector = embedding_model.encode([query], convert_to_tensor=False)
        query_vector = query_vector.astype('float32')
        faiss.normalize_L2(query_vector)
        
        if filter_filename or filter_page:
            valid_indices = [
                i for i, m in enumerate(st.session_state.metadata)
                if (filter_filename is None or m["filename"] == filter_filename) and
                   (filter_page is None or m["page"] == filter_page)
            ]
            
            if not valid_indices:
                return []
            
            filtered_embeddings = np.array([st.session_state.doc_embeddings[i] for i in valid_indices]).astype('float32')
            temp_index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
            faiss.normalize_L2(filtered_embeddings)
            temp_index.add(filtered_embeddings)
            
            D, I = temp_index.search(query_vector, k=min(k, len(valid_indices)))
            results = [(valid_indices[local_idx], score) for score, local_idx in zip(D[0], I[0])]
        else:
            D, I = st.session_state.faiss_index.search(query_vector, k=k)
            results = [(idx, score) for idx, score in zip(I[0], D[0])]
        
        return results
    except Exception as e:
        st.error(f"❌ Search error: {e}")
        return []

def render_highlighted_page(pdf, page_num, search_text):
    """Render PDF page with highlighted search text"""
    try:
        page = pdf.load_page(page_num - 1)
        
        if search_text:
            text_instances = page.search_for(search_text[:150])
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.set_colors(stroke=(1, 1, 0))
                highlight.update()
        
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    except Exception as e:
        st.error(f"Error rendering page: {e}")
        return None

# -------------------- FRONTEND --------------------

# Check if documents are loaded
if not st.session_state.app_started and not st.session_state.pdf_store:
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">💬 DocuMind AI</div>
        <div class="hero-subtitle">Advanced Multi-PDF Q&A with Draggable Controls</div>
        <div class="hero-description">
            Transform your PDF documents into intelligent knowledge bases with AI-powered search, automatic summarization, and a fully draggable, resizable control panel
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Smart Search</div>
            <div class="feature-description">Semantic search with cosine similarity for precise, context-aware results.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Rich Statistics</div>
            <div class="feature-description">Comprehensive document analytics including pages, chunks, and processing metrics.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🖱️</div>
            <div class="feature-title">Draggable Panel</div>
            <div class="feature-description">Fully interactive control panel that can be dragged, resized, and customized.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📑</div>
            <div class="feature-title">Auto Processing</div>
            <div class="feature-description">Automatic document processing with real-time progress tracking and statistics.</div>
        </div>
        """, unsafe_allow_html=True)
    
    # # File Uploader (without the pink upload section)
    # st.markdown("<br><br>", unsafe_allow_html=True)
    # uploaded_files = st.file_uploader(
    #     "Choose PDF files",
    #     type=["pdf"],
    #     accept_multiple_files=True,
    #     help="Upload one or more PDF documents",
    #     label_visibility="collapsed"
    # )
    
    
    # if uploaded_files:
    #     st.session_state.app_started = True
    #     st.rerun()
    
    # Start Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start Exploring Documents", use_container_width=True):
            st.session_state.app_started = True
            st.rerun()

else:
    # Main App Interface
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 3rem; margin-bottom: 10px;">💬 DocuMind AI</h1>
        <p style="font-size: 1.2rem; color: rgba(255,255,255,0.7);">Multi-PDF Q&A with RAG</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload Section - NOW SHOWN ON SECOND PAGE
    st.markdown("""
    <div class="upload-section">
        <div class="upload-title">👆 Drag & Drop Your PDFs</div>
        <div class="upload-subtitle">Upload PDF files below for automatic processing and intelligent analysis</div>
        <div class="upload-hint">💡 The control panel shows real-time statistics and can be moved around freely</div>
    </div>
    """, unsafe_allow_html=True)
    
    # File Upload Section
    uploaded_files = st.file_uploader(
        "📁 Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF documents"
    )
    
    if uploaded_files:
        current_files = [f.name for f in uploaded_files]
        
        if st.session_state.last_uploaded_files != current_files:
            st.session_state.doc_chunks = []
            st.session_state.doc_embeddings = []
            st.session_state.metadata = []
            st.session_state.pdf_store = {}
            st.session_state.pdf_summaries = {}
            st.session_state.faiss_index = None
            st.session_state.last_uploaded_files = current_files

            progress_bar = st.progress(0)
            status_text = st.empty()

            total_pages = 0
            for idx, file in enumerate(uploaded_files):
                status_text.markdown(f"**📄 Processing {file.name}...**")
                file.seek(0)
                summary, pages = extract_text_from_pdf(file, file.name)
                total_pages += pages
                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.markdown("**🔍 Building search index...**")
            if build_faiss_index():
                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)!")
    
    # Statistics Cards
    if st.session_state.pdf_store:
        st.markdown("### 📊 Document Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(st.session_state.pdf_store)}</div>
                <div class="stats-label">Documents</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_pages = sum(len(pdf) for pdf in st.session_state.pdf_store.values())
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{total_pages}</div>
                <div class="stats-label">Pages</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(st.session_state.doc_chunks)}</div>
                <div class="stats-label">Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            status = "✅ Ready" if st.session_state.faiss_index else "❌ Not Built"
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number" style="font-size: 2rem;">{status}</div>
                <div class="stats-label">Index Status</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabs for different query modes
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "📄 Page Query",
            "📚 Document Query",
            "🔍 Global Search"
        ])
        
        with tab1:
            st.markdown("### 📑 Document Summaries")
            for pdf_name, summary in st.session_state.pdf_summaries.items():
                with st.expander(f"📄 {pdf_name}", expanded=False):
                    st.markdown(summary)
        
        with tab2:
            st.markdown("### 📄 Query Specific Page")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                selected_pdf = st.selectbox(
                    "Select Document:",
                    list(st.session_state.pdf_store.keys()),
                    key="page_pdf"
                )
            
            if selected_pdf:
                pdf = st.session_state.pdf_store[selected_pdf]
                total_pages = len(pdf)
                
                with col2:
                    page_num = st.number_input(
                        "Page Number:",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        key="page_num"
                    )
                
                page = pdf.load_page(page_num - 1)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                st.image(img, caption=f"{selected_pdf} - Page {page_num}", use_container_width=True)
                
                query = st.text_input("💬 Ask about this page:", key="page_query")
                
                if query:
                    with st.spinner("🔍 Searching..."):
                        results = semantic_search(query, k=3, filter_filename=selected_pdf, filter_page=page_num)
                    
                    if results:
                        st.markdown("### ✅ Results")
                        for rank, (idx, score) in enumerate(results):
                            chunk = st.session_state.doc_chunks[idx]
                            meta = st.session_state.metadata[idx]
                            
                            st.markdown(f"**Match {rank+1}** (Score: {score:.3f})")
                            st.info(chunk)
                            
                            highlighted_img = render_highlighted_page(pdf, page_num, chunk[:100])
                            if highlighted_img:
                                st.image(highlighted_img, use_container_width=True)
                            st.markdown("---")
                    else:
                        st.warning("⚠️ No relevant text found.")
        
        with tab3:
            st.markdown("### 📚 Query Entire Document")
            selected_pdf = st.selectbox(
                "Select Document:",
                list(st.session_state.pdf_store.keys()),
                key="doc_pdf"
            )
            
            if selected_pdf:
                query = st.text_input("💬 Ask about this document:", key="doc_query")
                
                if query:
                    with st.spinner("🔍 Searching..."):
                        results = semantic_search(query, k=5, filter_filename=selected_pdf)
                    
                    if results:
                        st.markdown("### ✅ Top Results")
                        for rank, (idx, score) in enumerate(results):
                            chunk = st.session_state.doc_chunks[idx]
                            meta = st.session_state.metadata[idx]
                            
                            with st.expander(f"Result {rank+1} - Page {meta['page']} (Score: {score:.3f})", expanded=(rank==0)):
                                st.info(chunk)
                                
                                pdf = st.session_state.pdf_store[meta['filename']]
                                highlighted_img = render_highlighted_page(pdf, meta['page'], chunk[:150])
                                if highlighted_img:
                                    st.image(highlighted_img, use_container_width=True)
                    else:
                        st.warning("⚠️ No relevant results found.")
        
        with tab4:
            st.markdown("### 🔍 Search Across All Documents")
            query_all = st.text_input("💬 Ask a question:", key="global_query")
            
            if query_all:
                with st.spinner("🔍 Searching..."):
                    results = semantic_search(query_all, k=7)
                
                if results:
                    st.markdown(f"### ✅ Found {len(results)} Results")
                    
                    results_by_doc = {}
                    for idx, score in results:
                        meta = st.session_state.metadata[idx]
                        doc_name = meta['filename']
                        if doc_name not in results_by_doc:
                            results_by_doc[doc_name] = []
                        results_by_doc[doc_name].append((idx, score, meta))
                    
                    for doc_name, doc_results in results_by_doc.items():
                        st.markdown(f"#### 📄 {doc_name}")
                        
                        for i, (idx, score, meta) in enumerate(doc_results):
                            chunk = st.session_state.doc_chunks[idx]
                            
                            with st.expander(f"Match {i+1} - Page {meta['page']} (Score: {score:.3f})", expanded=(i==0)):
                                st.success(chunk)
                                
                                pdf = st.session_state.pdf_store[meta['filename']]
                                highlighted_img = render_highlighted_page(pdf, meta['page'], chunk[:150])
                                if highlighted_img:
                                    st.image(highlighted_img, use_container_width=True)
                        st.markdown("---")
                else:
                    st.warning("⚠️ No relevant results found.")
