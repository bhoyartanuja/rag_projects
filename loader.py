from pypdf import PdfReader

def load_and_chunk(pdf_path, chunk_size=500, overlap=50):
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for page in reader.pages:
        full_text += page.extract_text()
    
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += chunk_size - overlap
    
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk("sample_pdf_adobe.pdf")
    print(f"Total chunks: {len(chunks)}")
    print(f"\nFirst chunk:\n{chunks[0]}")
