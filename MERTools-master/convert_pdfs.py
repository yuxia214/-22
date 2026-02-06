import fitz # PyMuPDF
import os

files = [
    "/root/autodl-tmp/MERTools-master/空间状态模型.pdf",
    "/root/autodl-tmp/MERTools-master/缺失模态.pdf",
    "/root/autodl-tmp/MERTools-master/特征分解.pdf",
    "/root/autodl-tmp/MERTools-master/知识引导.pdf"
]

def convert_pdf_to_md(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    try:
        doc = fitz.open(pdf_path)
        text_content = []
        # Add filename as title
        filename = os.path.basename(pdf_path)
        text_content.append(f"# {filename}\n")
        
        for page_num, page in enumerate(doc):
            # Use "text" for plain text, or "markdown" if supported by newer PyMuPDF versions, 
            # but "text" is safer for basic extraction. 
            # Some versions support page.get_text("markdown") but let's stick to standard text for reliability.
            text = page.get_text()
            text_content.append(f"## Page {page_num + 1}\n\n{text}")
        
        md_content = "\n\n".join(text_content)
        
        output_path = pdf_path.rsplit('.', 1)[0] + ".md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        print(f"Converted {pdf_path} to {output_path}")
    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")

if __name__ == "__main__":
    for f in files:
        convert_pdf_to_md(f)
