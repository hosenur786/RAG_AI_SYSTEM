import pypdf
import re

class PdfReader:
    
    def __init__(self, path="./pdfs/2025-q1-earnings-transcript.pdf"):
        '''
        
        '''
        self.reader = pypdf.PdfReader(path)
        self.pages_text = ""
    
    def extract_text(self):
        '''
        Extracts text from the PDF file and stores it in the pages_text attribute.
        '''
        all_pages_text = []
        for _, page in enumerate(self.reader.pages):
            page_text = page.extract_text()
            if page_text:
                all_pages_text.append(page_text)
                
        # Join the text from all pages and normalize whitespace
        pdf_text = "\n".join(all_pages_text)
        pdf_text = re.sub(r'\n([ \t]*\n)?[ \t]{2,}', '¶', pdf_text)     # 1. mark paragraph breaks BEFORE space normalization
        pdf_text = re.sub(r' +', ' ', pdf_text)                         # 2. collapse multiple spaces
        pdf_text = re.sub(r'[ \t]*\n[ \t]*', ' ', pdf_text)             # 3. collapse word-wrap newlines → space
        pdf_text = pdf_text.replace('¶', '\n')                          # 4. restore paragraph breaks as \n
        pdf_text = re.sub(r' +', ' ', pdf_text).strip()                 # 5. final space cleanup
        print(f"Successfully extracted text. Total characters: {len(pdf_text)}")
        self.pages_text = pdf_text
        return pdf_text
                
    def extract_small_portion_of_the_pdf(self, min=0, max=None): 
        '''
        '''
        if self.pages_text == "":
            self.extract_text()
        return self.pages_text[min:max]
    
    def get_paragraphs(self, chunk_size=1200):
            if self.pages_text == "":
                self.extract_text()

            paragraphs = [
                p.strip()
                for p in self.pages_text.split('\n')
                if p.strip()
            ]

            chunks = []
            current_chunk = ""

            for paragraph in paragraphs:

                if len(current_chunk) + len(paragraph) < chunk_size:
                    current_chunk += " " + paragraph
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph

            if current_chunk:
                chunks.append(current_chunk.strip())

            print(f"Created {len(chunks)} larger chunks")
            return chunks
        
 #### testing ground
if __name__ == "__main__":
    pdf_reader = PdfReader()
    pdf_reader.extract_text()
    print(pdf_reader.extract_small_portion_of_the_pdf(min=0, max=100))
    print(pdf_reader.get_paragraphs()[:3])
        