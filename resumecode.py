import streamlit as st
import os
from io import BytesIO
import pypdf
import pdfplumber
from docx import Document
import nltk
import openai
from datetime import datetime
import json

def init_nltk():
    """Initialize NLTK data downloads with quiet mode"""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

# Initialize NLTK data
init_nltk()

class ResumeAnalyzer:
    def __init__(self):
        self.client = None
        if os.getenv('OPENAI_API_KEY'):
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def extract_text_from_pdf(self, file_bytes):
        """Extract text from PDF using both pypdf and pdfplumber for best results"""
        text = ""
        
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            st.warning(f"pdfplumber extraction failed: {e}")
        
        # If pdfplumber yielded little/no text, try pypdf as fallback
        if len(text.strip()) < 50:
            try:
                pdf_reader = pypdf.PdfReader(BytesIO(file_bytes))
                fallback_text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:  # Guard against None
                        fallback_text += page_text + "\n"
                
                # Use fallback if it's better
                if len(fallback_text.strip()) > len(text.strip()):
                    text = fallback_text
                    
            except Exception as e2:
                if not text.strip():  # Only error if no text at all
                    st.error(f"PDF text extraction failed: {e2}")
                    return ""
        
        if not text.strip():
            st.warning("No text could be extracted from this PDF. It may be a scanned document that requires OCR.")
        
        return text.strip()
    
    def extract_text_from_docx(self, file_bytes):
        """Extract text from DOCX file"""
        try:
            doc = Document(BytesIO(file_bytes))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error extracting text from DOCX: {e}")
            return ""
    
    def extract_text_from_file(self, uploaded_file):
        """Extract text from uploaded file based on file type"""
        file_bytes = uploaded_file.read()
        
        if uploaded_file.name.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_bytes)
        elif uploaded_file.name.lower().endswith('.docx'):
            return self.extract_text_from_docx(file_bytes)
        else:
            st.error("Unsupported file type. Please upload PDF or DOCX files only.")
            return ""
    
    def analyze_resume_with_ai(self, resume_text):
        """Analyze resume using OpenAI API"""
        if not self.client:
            return self.analyze_resume_basic(resume_text)
        
        try:
            # Truncate text if too long (roughly 8000 chars to stay under token limits)
            truncated_text = resume_text[:8000]
            if len(resume_text) > 8000:
                truncated_text += "\n[Content truncated for analysis]"
            
            prompt = f"""
            Analyze the following resume and provide detailed feedback. Please evaluate:
            
            1. Overall structure and formatting
            2. Content quality and relevance
            3. Skills and experience alignment
            4. Areas for improvement
            5. Missing sections or information
            6. Overall score (1-10)
            
            Resume text:
            {truncated_text}
            
            Please provide your analysis in a structured format with specific recommendations.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert HR professional and resume reviewer. Provide constructive, actionable feedback."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            st.error(f"Error analyzing resume with AI: {e}")
            return self.analyze_resume_basic(resume_text)
    
    def analyze_resume_basic(self, resume_text):
        """Basic resume analysis without AI"""
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        import re
        
        # Basic text analysis
        sentences = sent_tokenize(resume_text)
        words = word_tokenize(resume_text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        
        # Look for common resume sections
        sections = {
            'contact_info': bool(re.search(r'(email|phone|address|linkedin)', resume_text, re.IGNORECASE)),
            'experience': bool(re.search(r'(experience|work|employment|position)', resume_text, re.IGNORECASE)),
            'education': bool(re.search(r'(education|degree|university|college)', resume_text, re.IGNORECASE)),
            'skills': bool(re.search(r'(skills|technical|programming|software)', resume_text, re.IGNORECASE)),
            'summary': bool(re.search(r'(summary|objective|profile)', resume_text, re.IGNORECASE))
        }
        
        # Basic scoring
        score = 5  # Base score
        if sections['contact_info']: score += 1
        if sections['experience']: score += 2
        if sections['education']: score += 1
        if sections['skills']: score += 1
        
        # Word count analysis
        word_count = len(filtered_words)
        if 200 <= word_count <= 600:
            score += 1
        elif word_count < 100:
            score -= 1
        
        analysis = f"""
        **Basic Resume Analysis**
        
        **Text Statistics:**
        - Word count: {word_count}
        - Sentence count: {len(sentences)}
        
        **Sections Found:**
        - Contact Information: {'✓' if sections['contact_info'] else '✗'}
        - Work Experience: {'✓' if sections['experience'] else '✗'}
        - Education: {'✓' if sections['education'] else '✗'}
        - Skills: {'✓' if sections['skills'] else '✗'}
        - Summary/Objective: {'✓' if sections['summary'] else '✗'}
        
        **Overall Score: {min(score, 10)}/10**
        
        **Recommendations:**
        """
        
        if not sections['contact_info']:
            analysis += "\n- Add clear contact information (email, phone, LinkedIn)"
        if not sections['experience']:
            analysis += "\n- Include work experience or relevant projects"
        if not sections['education']:
            analysis += "\n- Add education background"
        if not sections['skills']:
            analysis += "\n- List relevant skills and technologies"
        if word_count < 200:
            analysis += "\n- Expand content - resume seems too brief"
        elif word_count > 600:
            analysis += "\n- Consider condensing content - resume may be too long"
        
        analysis += "\n\n*Note: For more detailed AI-powered analysis, please provide an OpenAI API key.*"
        
        return analysis

def main():
    st.set_page_config(
        page_title="AI Resume Analyzer",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🤖 AI Resume Analyzer")
    st.markdown("Upload your resume and get instant AI-powered feedback to improve your job applications!")
    
    # Initialize analyzer
    analyzer = ResumeAnalyzer()
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "OpenAI API Key (optional)", 
            type="password",
            help="For enhanced AI analysis. Leave blank for basic analysis."
        )
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            analyzer.client = openai.OpenAI(api_key=api_key)
        
        st.markdown("---")
        st.markdown("**Supported formats:** PDF, DOCX")
        st.markdown("**Max file size:** 10MB")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'docx'],
            help="Upload a PDF or DOCX file containing your resume (max 10MB)"
        )
        
        if uploaded_file is not None:
            # Check file size (10MB limit)
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 10:
                st.error(f"File too large ({file_size_mb:.1f}MB). Please upload a file smaller than 10MB.")
                return
            
            # Validate file type by extension
            if not (uploaded_file.name.lower().endswith('.pdf') or uploaded_file.name.lower().endswith('.docx')):
                st.error("Invalid file type. Please upload only PDF or DOCX files.")
                return
            
            st.success(f"File uploaded: {uploaded_file.name} ({file_size_mb:.1f}MB)")
            
            # Extract text
            with st.spinner("Extracting text from resume..."):
                resume_text = analyzer.extract_text_from_file(uploaded_file)
            
            if resume_text:
                st.subheader("📝 Extracted Text Preview")
                st.text_area("Resume Content", resume_text[:500] + "..." if len(resume_text) > 500 else resume_text, height=200)
                
                # Analysis button
                if st.button("🔍 Analyze Resume", type="primary"):
                    with st.spinner("Analyzing your resume... This may take a moment."):
                        analysis = analyzer.analyze_resume_with_ai(resume_text)
                    
                    # Store analysis in session state
                    st.session_state['analysis'] = analysis
                    st.session_state['resume_text'] = resume_text
                    st.session_state['filename'] = uploaded_file.name
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'analysis' in st.session_state:
            st.subheader(f"Analysis for: {st.session_state.get('filename', 'Unknown')}")
            st.markdown(st.session_state['analysis'])
            
            # Download option
            st.subheader("💾 Download Report")
            report_data = {
                'filename': st.session_state.get('filename', 'Unknown'),
                'analysis_date': datetime.now().isoformat(),
                'analysis': st.session_state['analysis'],
                'word_count': len(st.session_state.get('resume_text', '').split())
            }
            
            st.download_button(
                label="📥 Download Analysis Report",
                data=json.dumps(report_data, indent=2),
                file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.info("Upload a resume and click 'Analyze Resume' to see detailed feedback here.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with Streamlit • AI-powered resume analysis to help you land your dream job!</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()