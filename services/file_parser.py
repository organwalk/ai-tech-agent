import os
import io
import requests
import pdfplumber
import docx
from typing import Tuple


class FileParser:
    @staticmethod
    def download_file(file_url: str, timeout: int = 60) -> Tuple[bytes, str]:
        resp = requests.get(file_url, timeout=timeout)
        resp.raise_for_status()
        content_type = resp.headers.get('Content-Type', '').lower()
        return resp.content, content_type

    @staticmethod
    def extract_text(content: bytes, content_type: str, file_url: str) -> str:
        file_ext = os.path.splitext(file_url.split('?')[0])[1].lower()
        text_content = ""
        
        if 'pdf' in content_type or file_ext == '.pdf':
            print("识别为 PDF 文件，开始提取文本...")
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages_text = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
                text_content = "\n".join(pages_text)
        
        elif 'wordprocessingml' in content_type or file_ext == '.docx':
            print("识别为 Word(DOCX) 文件，开始提取文本...")
            doc = docx.Document(io.BytesIO(content))
            text_content = "\n".join([para.text for para in doc.paragraphs])
        
        else:
            print("识别为纯文本/Markdown 文件，直接读取...")
            text_content = content.decode('utf-8', errors='ignore')
        
        if not text_content.strip():
            raise ValueError("提取到的文件内容为空，可能是扫描版PDF或不支持的格式")
        
        return text_content
