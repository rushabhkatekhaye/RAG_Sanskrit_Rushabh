import os
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Import Ollama helper
try:
    from ollama_helper import OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ ollama_helper.py not found - using extractive QA only")

class SanskritRAGSystem:
    def __init__(self, data_dir="data", cache_dir="cache", use_ollama=True, ollama_model="phi"):
        """Initialize the RAG system with CPU-optimized models for Sanskrit
        
        Args:
            use_ollama: If True, uses Ollama for generation. If False, uses extractive QA
            ollama_model: Ollama model to use (phi, gemma:2b, etc.)
        """
        print("🚀 Initializing Sanskrit RAG System...")
        
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup generation method
        self.use_ollama = use_ollama and OLLAMA_AVAILABLE
        self.ollama_client = None
        
        if self.use_ollama:
            print("📌 Using Ollama for generation (local LLM)")
            self.ollama_client = OllamaClient(model=ollama_model)
            
            # Test Ollama connection
            if not self.ollama_client.test_connection():
                print("⚠️ Falling back to extractive QA")
                self.use_ollama = False
        else:
            print("📌 Using Extractive QA approach")
        
        # Load embedding model
        print("\n📥 Loading embedding model (multilingual - supports Sanskrit)...")
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.embedding_model.to('cpu')
        print("✅ Embedding model loaded on CPU!")
        
        self.chunks = []
        self.embeddings = None
        
    def load_documents(self, file_path: str) -> str:
        """Load Sanskrit documents from text file"""
        print(f"\n📄 Loading Sanskrit document: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✅ Loaded {len(content)} characters")
            print(f"✅ Document contains Sanskrit text in Devanagari script")
            return content
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        """Split Sanskrit text into overlapping chunks"""
        print(f"\n✂️ Chunking Sanskrit text...")
        print(f"   Chunk size: {chunk_size} characters")
        print(f"   Overlap: {overlap} characters")
        
        # Split by double newlines (paragraphs/stories) first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph doesn't exceed chunk size
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk
                # If single paragraph is too large, split it
                if len(para) > chunk_size:
                    # Split large paragraph into sentences (by । or ॥)
                    sentences = para.replace('॥', '।').split('।')
                    temp_chunk = ""
                    for sent in sentences:
                        if len(temp_chunk) + len(sent) < chunk_size:
                            temp_chunk += sent + "।"
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sent + "।"
                    current_chunk = temp_chunk
                else:
                    current_chunk = para + "\n\n"
        
        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        print(f"✅ Created {len(chunks)} chunks from Sanskrit text")
        
        # Show sample chunk
        if chunks:
            print(f"\n📝 Sample chunk (first 150 chars):")
            print(f"   {chunks[0][:150]}...")
        
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for Sanskrit text chunks"""
        print(f"\n🔢 Creating embeddings for {len(chunks)} Sanskrit chunks...")
        print("   Using multilingual model that understands Sanskrit...")
        
        embeddings = self.embedding_model.encode(
            chunks, 
            show_progress_bar=True,
            batch_size=8  # Smaller batch for CPU
        )
        
        print(f"✅ Embeddings created successfully!")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Each chunk is represented as a {embeddings.shape[1]}-dimensional vector")
        
        return embeddings
    
    def save_index(self):
        """Save chunks and embeddings to disk"""
        print("\n💾 Saving index to cache...")
        cache_file = self.cache_dir / "rag_index.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings
            }, f)
        print(f"✅ Index saved to {cache_file}")
        print(f"   Next run will be much faster (loads from cache)!")
    
    def load_index(self):
        """Load chunks and embeddings from disk"""
        cache_file = self.cache_dir / "rag_index.pkl"
        if cache_file.exists():
            print("\n📂 Loading cached index...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
            print(f"✅ Loaded {len(self.chunks)} chunks from cache")
            print(f"✅ Embeddings shape: {self.embeddings.shape}")
            return True
        return False
    
    def index_documents(self, file_path: str, force_reindex: bool = False):
        """Index Sanskrit documents"""
        if not force_reindex and self.load_index():
            print("✅ Using cached index (fast load)")
            return
        
        print("\n" + "="*70)
        print("📚 INDEXING SANSKRIT DOCUMENTS")
        print("="*70)
        
        # Load and chunk documents
        content = self.load_documents(file_path)
        self.chunks = self.chunk_text(content)
        
        # Create embeddings
        self.embeddings = self.create_embeddings(self.chunks)
        
        # Save for future use
        self.save_index()
        
        print("\n" + "="*70)
        print("✅ INDEXING COMPLETE!")
        print("="*70 + "\n")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        print(f"\n🔍 Searching for relevant Sanskrit text...")
        print(f"   Query: '{query}'")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        print(f"\n📊 Top {top_k} most relevant chunks:")
        for i, idx in enumerate(top_indices, 1):
            results.append({
                'chunk': self.chunks[idx],
                'similarity': float(similarities[idx]),
                'index': int(idx)
            })
            print(f"   {i}. Chunk #{idx} | Similarity: {similarities[idx]:.3f} ({similarities[idx]*100:.1f}%)")
        
        return results
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using Ollama or extractive QA"""
        
        # Get context
        best_chunk = context_chunks[0]['chunk']
        best_similarity = context_chunks[0]['similarity']
        
        # Use Ollama if available
        if self.use_ollama and self.ollama_client:
            return self._generate_with_ollama(query, context_chunks)
        else:
            return self._generate_extractive(query, context_chunks)
    
    def _generate_with_ollama(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using Ollama"""
        print(f"\n🤖 Generating answer using Ollama...")
        
        # Prepare context
        context = "\n\n".join([chunk['chunk'][:400] for chunk in context_chunks[:2]])
        
        # Create prompt
        prompt = f"""Based on the following Sanskrit text, answer the question clearly and concisely.

Sanskrit Text Context:
{context}

Question: {query}

Provide a clear answer in English. If the text contains English explanations, use them. Keep the answer focused and relevant.

Answer:"""
        
        print("   Generating response (5-10 seconds)...")
        
        # Generate with Ollama
        answer = self.ollama_client.generate(prompt, max_tokens=250, temperature=0.7)
        
        if answer and len(answer) > 20:
            # Add source info
            answer += f"\n\n[Source: Retrieved from Sanskrit text with {context_chunks[0]['similarity']:.1%} similarity]"
            print("✅ Answer generated successfully!")
            return answer
        else:
            print("⚠️ Ollama generation failed, using extractive QA...")
            return self._generate_extractive(query, context_chunks)
    
    def _generate_extractive(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using smart extractive QA with better formatting"""
        print(f"\n📝 Creating answer from retrieved Sanskrit text...")
        
        best_chunk = context_chunks[0]['chunk']
        best_similarity = context_chunks[0]['similarity']
        
        print(f"   Using top match (similarity: {best_similarity:.1%})...")
        
        # Determine query language
        is_english_query = any(char.isalpha() and ord(char) < 128 for char in query)
        
        # Parse chunk
        lines = [l.strip() for l in best_chunk.split('\n') if l.strip()]
        
        # Extract different content types
        sanskrit_lines = []
        english_lines = []
        title = ""
        
        for i, line in enumerate(lines):
            # Title detection
            if i < 3 and (line.startswith('**') or line.startswith('#')):
                title = line.replace('**', '').replace('#', '').strip()
                continue
            
            # Check content type
            has_devanagari = any('\u0900' <= c <= '\u097F' for c in line)
            
            if has_devanagari:
                sanskrit_lines.append(line)
            elif len(line) > 30 and not line.startswith(('*', '#', '-')):
                english_lines.append(line)
        
        # Extract Sanskrit sentences
        sanskrit_text = ' '.join(sanskrit_lines)
        sanskrit_sentences = [s.strip() for s in sanskrit_text.replace('॥', '।').split('।') 
                             if len(s.strip()) > 15]
        
        # Build answer
        answer = []
        
        # Add story title
        if title:
            answer.append(f"📖 **{title}**\n")
        
        # For English queries - provide readable summary
        if is_english_query:
            # Extract English content
            if english_lines:
                # Find the most informative English sentence
                best_english = max(english_lines, key=len)
                answer.append("**Summary:**")
                answer.append(best_english[:400] + ("..." if len(best_english) > 400 else ""))
                answer.append("")
            
            # Add Sanskrit excerpt
            if sanskrit_sentences:
                answer.append("**Sanskrit Excerpt:**")
                excerpt = ' । '.join(sanskrit_sentences[:3]) + ' ।'
                answer.append(excerpt)
                answer.append("")
        
        # For Sanskrit queries
        else:
            # Provide Sanskrit content first
            if sanskrit_sentences:
                answer.append("**संस्कृत कथा (Sanskrit Story):**")
                content = ' । '.join(sanskrit_sentences[:5]) + ' ।'
                answer.append(content)
                answer.append("")
            
            # Add English explanation if available
            if english_lines:
                answer.append("**English Translation/Context:**")
                answer.append(english_lines[0][:350])
                answer.append("")
        
        # Add all relevant chunks preview
        if len(context_chunks) > 1:
            answer.append("**Related Content:**")
            for i, chunk in enumerate(context_chunks[:2], 1):
                # Get first line of chunk
                first_line = chunk['chunk'].split('\n')[0].strip()[:100]
                if first_line:
                    answer.append(f"{i}. {first_line}... (similarity: {chunk['similarity']:.1%})")
            answer.append("")
        
        # Source metadata
        answer.append(f"_[Source: Chunk #{context_chunks[0]['index']} | Match: {best_similarity:.1%}]_")
        
        result = '\n'.join(answer)
        print("✅ Answer created successfully!")
        return result
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """Main query function - retrieve and generate"""
        print("\n" + "="*70)
        print("💬 PROCESSING QUERY")
        print("="*70)
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k)
        
        # Generate answer
        answer = self.generate_answer(question, relevant_chunks)
        
        result = {
            'question': question,
            'answer': answer,
            'relevant_chunks': relevant_chunks
        }
        
        print("\n" + "="*70)
        print("✅ QUERY PROCESSING COMPLETE")
        print("="*70)
        
        return result


def main():
    """Main function to run the Sanskrit RAG system"""
    print("\n" + "🕉️ "*30)
    print("       SANSKRIT DOCUMENT RAG SYSTEM")
    print("        (Ollama + Extractive QA)")
    print("           CPU-Only | Fully Local")
    print("🕉️ "*30 + "\n")
    
    # Initialize system (use_ollama=True to enable Ollama)
    # Try gemma:2b first, falls back to extractive if not available
    rag = SanskritRAGSystem(use_ollama=True, ollama_model="gemma:2b")
    
    # Index documents
    data_file = "data/sanskrit_docs.txt"
    
    if not os.path.exists(data_file):
        print(f"\n❌ ERROR: File not found: {data_file}")
        print("\n📋 Please follow these steps:")
        print("   1. Create a 'data' folder in your project directory")
        print("   2. Create 'sanskrit_docs.txt' file inside 'data' folder")
        print("   3. Copy your Sanskrit text from Rag-docs.docx into this file")
        return
    
    # Index documents (set force_reindex=True to rebuild index)
    rag.index_documents(data_file, force_reindex=False)
    
    # Interactive query loop
    print("\n" + "="*70)
    print("🎯 SYSTEM READY FOR QUERIES!")
    print("="*70)
    print("\n📝 You can ask questions in:")
    print("   • English (e.g., 'Who is Shankhanaad?')")
    print("   • Sanskrit (e.g., 'मूर्खभृत्यस्य कथा किम् अस्ति?')")
    print("   • Transliterated Sanskrit")
    print("\n💡 Type 'exit' or 'quit' to stop")
    print("="*70 + "\n")
    
    # Example queries to show
    print("💡 Example questions you can try:")
    print("   - What happened to Shankhanaad?")
    print("   - Tell me about Kalidas")
    print("   - What is the story of the old woman?")
    print("   - मूर्खभृत्यस्य कथा किम् अस्ति?")
    print()
    
    while True:
        query = input("\n❓ Enter your question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Thank you for using Sanskrit RAG System!")
            print("🙏 धन्यवादः!")
            break
        
        if not query:
            print("⚠️ Please enter a question")
            continue
        
        # Process query
        try:
            result = rag.query(query, top_k=3)
            
            # Display results
            print("\n" + "─"*70)
            print("📝 ANSWER:")
            print("─"*70)
            print(result['answer'])
            
            print("\n" + "─"*70)
            print("📚 RELEVANT SOURCES (Top matching Sanskrit text chunks):")
            print("─"*70)
            for i, chunk in enumerate(result['relevant_chunks'], 1):
                print(f"\n{i}. Similarity Score: {chunk['similarity']:.3f} ({chunk['similarity']*100:.1f}%)")
                print(f"   Chunk #{chunk['index']}")
                print(f"   Preview: {chunk['chunk'][:250]}...")
                print()
                
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    main()