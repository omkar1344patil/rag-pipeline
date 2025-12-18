"""
Standalone Hybrid Chunker Test
No dependencies on existing code
"""
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleHybridChunker:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded!\n")
    
    def parse_structure(self, text):
        """Find headers, sections, and structure"""
        lines = text.split('\n')
        structure = {
            'headers': [],
            'sections': [],
            'lists': [],
            'paragraphs': []
        }
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Markdown headers
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
            
            # List items
            elif stripped.startswith('-') or stripped.startswith('*') or re.match(r'^\d+\.', stripped):
                structure['lists'].append({
                    'line': i,
                    'text': line
                })
            
            # Regular paragraphs
            elif len(stripped) > 20:  # Meaningful content
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
        print(f"  Found {len(structure['lists'])} list items")
        print(f"  Found {len(structure['paragraphs'])} paragraphs")
        
        # Step 2: Identify structural chunks
        print("\n[Step 2] Creating structure-based chunks...")
        chunks = []
        
        # Group content by headers
        lines = text.split('\n')
        if structure['headers']:
            for i, header in enumerate(structure['headers']):
                start_line = header['line']
                end_line = structure['headers'][i + 1]['line'] if i + 1 < len(structure['headers']) else len(lines)
                
                section_text = '\n'.join(lines[start_line:end_line])
                
                chunk_info = {
                    'text': section_text,
                    'type': 'section',
                    'title': header['title'],
                    'level': header['level'],
                    'size': len(section_text)
                }
                
                print(f"  Section: '{header['title']}' ({len(section_text)} chars)")
                chunks.append(chunk_info)
        else:
            # No headers, treat as one section
            chunks.append({
                'text': text,
                'type': 'document',
                'title': 'Full Document',
                'level': 0,
                'size': len(text)
            })
        
        # Step 3: Apply semantic chunking to large sections
        print("\n[Step 3] Applying semantic chunking to large sections...")
        final_chunks = []
        
        for chunk in chunks:
            if chunk['size'] > 1000:  # Too large
                print(f"\n  '{chunk['title']}' is too large ({chunk['size']} chars)")
                print(f"  Applying semantic chunking...")
                
                semantic_chunks = self.semantic_split(chunk['text'], max_chunk_size=500)
                
                for j, sem_chunk in enumerate(semantic_chunks):
                    final_chunks.append({
                        'text': sem_chunk,
                        'type': 'semantic_subsection',
                        'parent_title': chunk['title'],
                        'parent_level': chunk['level'],
                        'chunk_index': j,
                        'size': len(sem_chunk)
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
    
    def compare_embeddings(self, text1, text2):
        """Compare two texts via embeddings"""
        print("\n" + "="*80)
        print("EMBEDDING COMPARISON")
        print("="*80)
        
        print(f"\nText 1: '{text1[:80]}...'")
        print(f"Text 2: '{text2[:80]}...'")
        
        # Create embeddings
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)
        
        # Calculate similarity
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        
        print(f"\nCosine Similarity: {similarity:.4f}")
        
        if similarity > 0.8:
            print("→ Very similar topics!")
        elif similarity > 0.6:
            print("→ Related topics")
        elif similarity > 0.4:
            print("→ Somewhat related")
        else:
            print("→ Different topics")
        
        # Show difference
        diff = np.abs(emb1 - emb2)
        print(f"\nAverage difference per dimension: {np.mean(diff):.4f}")
        print(f"Max difference: {np.max(diff):.4f}")
        
        return similarity


# ============================================================================
# TEST SAMPLE DOCUMENT
# ============================================================================

SAMPLE_DOC = """
# Big Data: A Survey

## 1. Introduction

Big data is a term that describes large volumes of data. The term has been in use since the 1990s, with some giving credit to John Mashey for popularizing the term. Big data usually includes data sets with sizes beyond the ability of commonly used software tools to capture, curate, manage, and process data within a tolerable elapsed time.

### 1.1 Background

Over the past 20 years, data has increased in a large scale in various fields. According to a report from International Data Corporation (IDC), in 2011, the overall created and copied data volume in the world was 1.8ZB, which increased by nearly nine times within five years.

The explosive increase of global data has become a major challenge for data storage and analysis. Traditional databases cannot handle such massive amounts of data efficiently.

### 1.2 Characteristics

Big data is characterized by the three Vs:
- Volume: The quantity of generated and stored data
- Velocity: The speed at which the data is generated
- Variety: The type and nature of the data

## 2. Data Storage

Data storage is a fundamental challenge in big data systems. Several approaches have been developed to address this challenge.

### 2.1 Distributed File Systems

Distributed file systems like HDFS (Hadoop Distributed File System) provide reliable storage for big data. HDFS was designed to store very large files with streaming data access patterns, running on clusters of commodity hardware.

The architecture of HDFS follows a master-slave pattern. The NameNode acts as the master server, managing the file system namespace and regulating access to files by clients.

### 2.2 NoSQL Databases

NoSQL databases provide mechanisms for storage and retrieval of data that is modeled differently from tabular relations used in relational databases. They are particularly useful for handling unstructured data.

## 3. Conclusion

Big data continues to evolve as a critical component of modern computing. Future research will focus on improving scalability and reducing processing time.
"""


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" HYBRID CHUNKING + EMBEDDING VISUALIZATION TEST")
    print("="*80)
    
    # Initialize
    chunker = SimpleHybridChunker()
    
    # Test 1: Hybrid Chunking
    chunks = chunker.hybrid_chunk(SAMPLE_DOC)
    
    print("\n" + "="*80)
    print("FINAL CHUNKS")
    print("="*80)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Type: {chunk['type']}")
        if 'title' in chunk:
            print(f"Title: {chunk['title']}")
        if 'parent_title' in chunk:
            print(f"Parent: {chunk['parent_title']}")
        print(f"Size: {chunk['size']} chars")
        print(f"Text preview: {chunk['text'][:150]}...")
    
    # Test 2: Visualize Embedding
    print("\n\n")
    sample_text = "Big data requires efficient storage systems like HDFS."
    embedding = chunker.visualize_embedding(sample_text)
    
    # Test 3: Compare Embeddings
    print("\n\n")
    text_a = "HDFS is a distributed file system for big data storage."
    text_b = "Hadoop Distributed File System stores large datasets."
    text_c = "Machine learning algorithms require training data."
    
    print("\n--- Comparing similar texts ---")
    chunker.compare_embeddings(text_a, text_b)
    
    print("\n--- Comparing different texts ---")
    chunker.compare_embeddings(text_a, text_c)
    
    print("\n\n" + "="*80)
    print(" TEST COMPLETE")
    print("="*80)