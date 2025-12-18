"""
Standalone Hybrid Chunker Test - Works with YOUR documents
"""
import re
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleHybridChunker:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded!\n")
    
    def load_document(self, filepath):
        """Load document from file"""
        print(f"Loading document: {filepath}")
        
        try:
            if filepath.endswith('.pdf'):
                # PDF support
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(filepath)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    print(f"  ✓ Loaded {len(reader.pages)} pages from PDF")
                except ImportError:
                    print("  ✗ pypdf not installed. Run: pip install pypdf")
                    return None
            else:
                # Text file (txt, md, etc.)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                print(f"  ✓ Loaded {len(text)} characters")
            
            return text
        except Exception as e:
            print(f"  ✗ Error loading file: {e}")
            return None
    
    def parse_structure(self, text):
        """Find headers, sections, and structure"""
        lines = text.split('\n')
        structure = {
            'headers': [],
            'sections': [],
            'lists': [],
            'paragraphs': [],
            'feedback_entries': []
        }
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Markdown headers (#, ##, ###)
            if stripped.startswith('#'):
                level = len(stripped) - len(stripped.lstrip('#'))
                title = stripped.lstrip('#').strip()
                structure['headers'].append({
                    'line': i,
                    'level': level,
                    'title': title,
                    'text': line
                })
            
            # Numbered sections (1.1, 2.3.4, etc.)
            elif re.match(r'^\d+\.\d+', stripped):
                structure['sections'].append({
                    'line': i,
                    'text': line
                })
            
            # Feedback pattern: "Feedback from Name (date):"
            elif re.match(r'^Feedback from .+ \(\d+/\d+', stripped, re.IGNORECASE):
                structure['feedback_entries'].append({
                    'line': i,
                    'text': line
                })
            
            # List items (-, *, or numbered)
            elif stripped.startswith('-') or stripped.startswith('*') or re.match(r'^\d+\.', stripped):
                structure['lists'].append({
                    'line': i,
                    'text': line
                })
            
            # Regular paragraphs (meaningful content)
            elif len(stripped) > 20:
                structure['paragraphs'].append({
                    'line': i,
                    'text': line
                })
        
        return structure
    
    def semantic_split(self, text, max_chunk_size=500):
        """Split text at semantic boundaries"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 2:
            return [text]
        
        print(f"  Found {len(sentences)} sentences")
        
        # Embed sentences
        print("  Computing sentence embeddings...")
        embeddings = self.model.encode(sentences)
        
        # Calculate similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        
        print(f"  Average similarity: {np.mean(similarities):.3f}")
        
        # Find split points (where similarity drops)
        threshold = np.mean(similarities) - 0.1  # Below average = topic shift
        split_points = [0]
        
        for i, sim in enumerate(similarities):
            if sim < threshold:
                print(f"  Topic shift at sentence {i+1} (similarity: {sim:.3f})")
                split_points.append(i + 1)
        
        split_points.append(len(sentences))
        
        # Create chunks
        chunks = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            chunk_text = ' '.join(sentences[start:end])
            
            # If chunk too large, force split
            if len(chunk_text) > max_chunk_size:
                print(f"  Chunk too large ({len(chunk_text)} chars), splitting...")
                mid = len(sentences[start:end]) // 2
                chunks.append(' '.join(sentences[start:start+mid]))
                chunks.append(' '.join(sentences[start+mid:end]))
            else:
                chunks.append(chunk_text)
        
        return chunks
    
    def extract_metadata_from_text(self, text):
        """Extract metadata like dates, names from text"""
        metadata = {}
        
        # Extract dates
        date_patterns = [
            r'\((\d{1,2}/\d{1,2})\)',  # (23/10)
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # 23-10-2024
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text[:200])  # Check first 200 chars
            if matches:
                metadata['date'] = matches[0]
                break
        
        # Extract names (after "from")
        name_match = re.search(r'from\s+([A-Z][a-z]+)', text[:100])
        if name_match:
            metadata['author'] = name_match.group(1)
        
        return metadata
    
    def hybrid_chunk(self, text):
        """Combine structure-aware + semantic chunking"""
        print("\n" + "="*80)
        print("HYBRID CHUNKING")
        print("="*80)
        
        # Step 1: Parse structure
        print("\n[Step 1] Parsing document structure...")
        structure = self.parse_structure(text)
        
        print(f"  Found {len(structure['headers'])} headers")
        print(f"  Found {len(structure['sections'])} numbered sections")
        print(f"  Found {len(structure['feedback_entries'])} feedback entries")
        print(f"  Found {len(structure['lists'])} list items")
        print(f"  Found {len(structure['paragraphs'])} paragraphs")
        
        # Step 2: Identify structural chunks
        print("\n[Step 2] Creating structure-based chunks...")
        chunks = []
        lines = text.split('\n')
        
        # Strategy: Group by feedback entries first (if any), then headers
        if structure['feedback_entries']:
            print("  Grouping by feedback entries...")
            for i, feedback in enumerate(structure['feedback_entries']):
                start_line = feedback['line']
                end_line = structure['feedback_entries'][i + 1]['line'] if i + 1 < len(structure['feedback_entries']) else len(lines)
                
                section_text = '\n'.join(lines[start_line:end_line])
                metadata = self.extract_metadata_from_text(section_text)
                
                chunk_info = {
                    'text': section_text,
                    'type': 'feedback',
                    'title': feedback['text'][:80],
                    'size': len(section_text),
                    'metadata': metadata
                }
                
                print(f"  Feedback: '{chunk_info['title']}' ({len(section_text)} chars)")
                if metadata:
                    print(f"    Metadata: {metadata}")
                chunks.append(chunk_info)
        
        elif structure['headers']:
            print("  Grouping by headers...")
            for i, header in enumerate(structure['headers']):
                start_line = header['line']
                end_line = structure['headers'][i + 1]['line'] if i + 1 < len(structure['headers']) else len(lines)
                
                section_text = '\n'.join(lines[start_line:end_line])
                
                chunk_info = {
                    'text': section_text,
                    'type': 'section',
                    'title': header['title'],
                    'level': header['level'],
                    'size': len(section_text),
                    'metadata': {}
                }
                
                print(f"  Section: '{header['title']}' ({len(section_text)} chars)")
                chunks.append(chunk_info)
        else:
            # No structure found, split into paragraphs
            print("  No clear structure, splitting by paragraphs...")
            paragraphs = text.split('\n\n')
            for i, para in enumerate(paragraphs):
                if len(para.strip()) > 50:
                    chunks.append({
                        'text': para,
                        'type': 'paragraph',
                        'title': f'Paragraph {i+1}',
                        'size': len(para),
                        'metadata': {}
                    })
        
        # Step 3: Apply semantic chunking to large sections
        print("\n[Step 3] Applying semantic chunking to large sections...")
        final_chunks = []
        
        for chunk in chunks:
            if chunk['size'] > 1000:  # Too large
                print(f"\n  '{chunk['title'][:50]}...' is too large ({chunk['size']} chars)")
                print(f"  Applying semantic chunking...")
                
                semantic_chunks = self.semantic_split(chunk['text'], max_chunk_size=500)
                
                for j, sem_chunk in enumerate(semantic_chunks):
                    final_chunks.append({
                        'text': sem_chunk,
                        'type': 'semantic_subsection',
                        'parent_title': chunk['title'],
                        'parent_type': chunk['type'],
                        'chunk_index': j,
                        'size': len(sem_chunk),
                        'metadata': chunk['metadata']
                    })
                    print(f"    → Sub-chunk {j+1}: {len(sem_chunk)} chars")
            else:
                # Keep as is
                final_chunks.append(chunk)
        
        print(f"\n[Result] Created {len(final_chunks)} final chunks")
        return final_chunks
    
    def visualize_embedding(self, text):
        """Show what an embedding looks like"""
        print("\n" + "="*80)
        print("EMBEDDING VISUALIZATION")
        print("="*80)
        
        print(f"\nText: '{text[:100]}...'")
        print(f"Length: {len(text)} characters\n")
        
        # Create embedding
        embedding = self.model.encode(text)
        
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"Embedding type: {type(embedding)}")
        print(f"Value range: [{embedding.min():.3f}, {embedding.max():.3f}]")
        
        print(f"\nFirst 10 dimensions:")
        for i in range(10):
            print(f"  Dim {i}: {embedding[i]:.6f}")
        
        print(f"\nLast 10 dimensions:")
        for i in range(-10, 0):
            print(f"  Dim {len(embedding)+i}: {embedding[i]:.6f}")
        
        # Show magnitude
        magnitude = np.linalg.norm(embedding)
        print(f"\nVector magnitude (length): {magnitude:.3f}")
        
        return embedding
    
    def save_chunks(self, chunks, output_file='chunks_output.txt'):
        """Save chunks to a file for inspection"""
        print(f"\nSaving chunks to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(" CHUNKING RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for i, chunk in enumerate(chunks):
                f.write(f"\n{'='*80}\n")
                f.write(f"CHUNK {i+1}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Type: {chunk['type']}\n")
                if 'title' in chunk:
                    f.write(f"Title: {chunk['title']}\n")
                if 'parent_title' in chunk:
                    f.write(f"Parent: {chunk['parent_title']}\n")
                if 'metadata' in chunk and chunk['metadata']:
                    f.write(f"Metadata: {chunk['metadata']}\n")
                f.write(f"Size: {chunk['size']} characters\n")
                f.write(f"\n{'-'*80}\n")
                f.write(f"CONTENT:\n")
                f.write(f"{'-'*80}\n")
                f.write(chunk['text'])
                f.write(f"\n\n")
        
        print(f"  ✓ Saved {len(chunks)} chunks")


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" HYBRID CHUNKING + EMBEDDING VISUALIZATION TEST")
    print("="*80)
    
    # Initialize
    chunker = SimpleHybridChunker()
    
    # Get file from command line or ask user
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        print("\nEnter the path to your document:")
        print("  (Supports: .txt, .md, .pdf)")
        filepath = input("  File path: ").strip()
    
    # Load document
    text = chunker.load_document(filepath)
    
    if not text:
        print("\n✗ Failed to load document. Exiting.")
        sys.exit(1)
    
    print(f"\nDocument stats:")
    print(f"  Total length: {len(text)} characters")
    print(f"  Lines: {len(text.split(chr(10)))}")
    print(f"  Words: ~{len(text.split())}")
    
    # Perform hybrid chunking
    chunks = chunker.hybrid_chunk(text)
    
    # Display results
    print("\n" + "="*80)
    print("FINAL CHUNKS SUMMARY")
    print("="*80)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Type: {chunk['type']}")
        if 'title' in chunk:
            print(f"Title: {chunk['title'][:60]}")
        if 'parent_title' in chunk:
            print(f"Parent: {chunk['parent_title'][:60]}")
        if 'metadata' in chunk and chunk['metadata']:
            print(f"Metadata: {chunk['metadata']}")
        print(f"Size: {chunk['size']} chars")
        print(f"Preview: {chunk['text'][:100]}...")
    
    # Save chunks to file
    output_file = filepath.rsplit('.', 1)[0] + '_chunks.txt'
    chunker.save_chunks(chunks, output_file)
    
    # Show embedding example from first chunk
    if chunks:
        print("\n\n")
        sample_text = chunks[0]['text'][:200]  # First 200 chars of first chunk
        embedding = chunker.visualize_embedding(sample_text)
    
    print("\n\n" + "="*80)
    print(" TEST COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
    print("You can inspect the chunks in that file!")