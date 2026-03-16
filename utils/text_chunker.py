import re
from typing import List


class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text_content: str) -> List[str]:
        chunks = []
        
        text_content = re.sub(r'\n{3,}', '\n\n', text_content)
        paragraphs = text_content.split('\n\n')
        
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_len = len(para)
            
            if para_len > self.chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                sentences = re.split(r'(?<=[。！？!?\n])', para)
                temp_chunk = ""
                for sent in sentences:
                    if len(temp_chunk) + len(sent) > self.chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        overlap_text = temp_chunk[-self.overlap:] if len(temp_chunk) > self.overlap else temp_chunk
                        match = re.search(r'[。！？!?\n、，,]', overlap_text)
                        if match and match.end() < len(overlap_text):
                            overlap_text = overlap_text[match.end():].strip()
                        temp_chunk = overlap_text + sent
                    else:
                        temp_chunk += sent
                
                if temp_chunk.strip():
                    chunks.append(temp_chunk.strip())
                continue
            
            if current_length + para_len + (2 if current_chunk else 0) > self.chunk_size:
                chunks.append("\n\n".join(current_chunk))
                last_chunk_text = "\n\n".join(current_chunk)
                overlap_text = last_chunk_text[-self.overlap:] if len(last_chunk_text) > self.overlap else last_chunk_text
                
                match = re.search(r'[。！？!?\n]', overlap_text)
                if match and match.end() < len(overlap_text):
                    overlap_text = overlap_text[match.end():].strip()
                
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_length = len(overlap_text) + para_len + (2 if overlap_text else 0)
            else:
                current_chunk.append(para)
                current_length += para_len + (2 if len(current_chunk) > 1 else 0)
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
