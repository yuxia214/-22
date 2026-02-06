
import fitz  # PyMuPDF
import os

def convert_pdf_to_md(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    md_path = pdf_path.rsplit('.', 1)[0] + '.md'
    
    try:
        doc = fitz.open(pdf_path)
        text_content = []
        
        for i, page in enumerate(doc):
            text = page.get_text()
            text_content.append(f"## Page {i+1}\n\n{text}\n")
            
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_content))
            
        print(f"Successfully converted {pdf_path} to {md_path}")
        
    except Exception as e:
        print(f"Error converting {pdf_path}: {str(e)}")

files = [
    "/root/autodl-tmp/MERTools-master/空间状态模型.pdf",
    "/root/autodl-tmp/MERTools-master/缺失模态.pdf",
    "/root/autodl-tmp/MERTools-master/特征分解.pdf",
    "/root/autodl-tmp/MERTools-master/知识引导.pdf"
]

for f in files:
    convert_pdf_to_md(f)
