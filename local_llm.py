# Let's import these libraries
import torch  # PyTorch, the backend for transformers
from transformers import AutoModelForCausalLM
from transformers import TextStreamer, TextIteratorStreamer
from transformers import AutoTokenizer
from threading import Thread
from pdf_reader import PdfReader
from local_embedding import LocalEmbedding
import os
from huggingface_hub import login


class AiModel():

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        '''
            initializing my AiModel class where we need the model name to create a tokenizer and a model
            the tokenizer will transform our text into numbers for our model to understand then will transform the numbers from the model to text so we understand it
            the model is the LLM that will think and give us the answers to our questions
        '''
        self.model_name = model_name
        print("running checks to make sure everything is good...")
        self.hugging_face_auth()
        self.hardware_check()
        print("we are creating the model this might take a while please wait...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code = True)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.model_name, torch_dtype=torch.float32)
        print("Model device:", next(self.model.parameters()).device)
    

    def hardware_check(self):
        '''
            making sure we are working on a local GPU rather than CPU to take advantage of Local LLMs
        '''
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: No GPU detected.")
    

    def hugging_face_auth(self):
        '''
            in order to download the right model to work on some of the model are gated by HuggingFace therefore we must authenticate first
        '''
        # getting token from .env
        HUGGING_FACE_TOKEN=os.environ.get("HF_TOKEN")

        # logging in
        print("Attempting Hugging Face login...")
        login(token=HUGGING_FACE_TOKEN)
        print("Login successful!")


    def ask_a_question(self, prompt="Hello there!"):
        '''
            formats the prompt as a chat message so the instruction-tuned model
            knows to respond rather than do raw text completion
        '''
        # wrap the prompt in the chat format the model was trained on
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # getting input and prompt
        inputs = self.tokenizer(formatted, return_tensors="pt")

        print("=" * 50)
        print("Input tokens:", inputs["input_ids"].shape[1])
        print("=" * 50)

        streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=120.0
            )

        # max_new_tokens caps the response length; streamer handles all printing internally.
        self.model.generate(**inputs, max_new_tokens=100, streamer=streamer)
    
    def ask_a_question_from_pdf(self, pdf_path, prompt="tell me what is this pdf about"):
        '''
            this function allows the user to take a pdf and ask some questions about the pdf, 
            performing RAG operation
        '''
        # creating our PdfReader to work with the pdf text
        pdf_reader = PdfReader(pdf_path)
        pdf_paragraphs = pdf_reader.get_paragraphs()
        
        # embedding and indexing all chunks
        local_embedding = LocalEmbedding()
        local_embedding.build_index(pdf_paragraphs)

        # getting relevant sections of the pdf
        relevent_sections = local_embedding.get_context(prompt, 10)

        # crafting message
        full_prompt_for_rag = self.full_prompt_for_rag(relevent_sections=relevent_sections, question_prompt=prompt)
        messages = [{"role": "user", "content": full_prompt_for_rag}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # getting input and prompt
        inputs = self.tokenizer(formatted, return_tensors="pt")

        # Streaming output — tokens are printed to the terminal as they are generated,
        # instead of waiting for the full response to be built in memory first.
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # max_new_tokens caps the response length; streamer handles all printing internally.
        self.model.generate(**inputs, max_new_tokens=100, streamer=streamer)


    def ask_a_question_from_pdf_stream(self, pdf_path: str, prompt: str = "tell me what is this pdf about", local_embedding=None):
        '''
            Streaming variant of ask_a_question_from_pdf.
            Yields decoded text chunks via TextIteratorStreamer so callers (e.g. st.write_stream)
            can consume tokens in real time without blocking on stdout.

            Args:
                pdf_path:        Path to the PDF on disk.
                prompt:          User question string.
                local_embedding: Pre-built LocalEmbedding instance (already indexed).
                                 If None, builds the index from scratch.
            Yields:
                str chunks as the model generates them.
        '''
        
        if local_embedding is None:
            pdf_reader = PdfReader(pdf_path)
            pdf_paragraphs = pdf_reader.get_paragraphs()
            local_embedding = LocalEmbedding()
            local_embedding.build_index(pdf_paragraphs)

        relevant_sections, sources = local_embedding.get_context(prompt, k=3)
        self.last_sources = sources
        print("\n" + "="*80)
        print("QUESTION:")
        print(prompt)

        print("\nRETRIEVED CONTEXT:")
        print(relevant_sections)

        print("="*80 + "\n")
        full_prompt = self.full_prompt_for_rag(
            relevent_sections=relevant_sections,
            question_prompt=prompt,
        )
        messages = [
                        {
                            "role": "system",
                            "content": "Answer only using the provided document context. If the answer is not present, say the document does not contain information on this topic."
                        },
                        {
                            "role": "user",
                            "content": f"""
                    Document Context:

                    {relevant_sections}

                    Question:
                    {prompt}
                    """
                        }
                    ]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt")

        print("=" * 50)
        print("Input tokens:", inputs["input_ids"].shape[1])
        print("=" * 50)
        # TextIteratorStreamer stores tokens in a Queue instead of printing to stdout.
        # timeout=30 prevents blocking forever if the generation thread crashes.
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120.0
        )

        # model.generate() is blocking — run it in a daemon thread so the main
        # thread can iterate the streamer queue without deadlocking.
        def safe_generate():
            try:
                print("=" * 50)
                print("Generation started")
                print("Calling model.generate()")

                self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    streamer=streamer
                )
                print("model.generate() returned")
                print("Generation finished")
                print("=" * 50)

            except Exception as e:
                import traceback
                print("=" * 50)
                print("GENERATION ERROR:")
                print(type(e).__name__)
                print(str(e))
                traceback.print_exc()
                print("=" * 50)

        thread = Thread(
            target=safe_generate,
            daemon=False
        )

        print("About to start thread")
        thread.start()
        print("Thread started")

        
        try:
            for chunk in streamer:
                yield chunk
        finally:
            thread.join()


    def full_prompt_for_rag(self, relevent_sections, question_prompt):
        '''
            this is a prompt constructor that will put together the user question, the pdf relevant sections, and system prompt
        '''
        return f"""
            <|system|>
                You are an AI assistant. Answer the following question based *only* on the provided document text. 
                If the answer is not found in the document, say "The document does not contain information on this topic." Do not use any prior knowledge.

                Document Text:
                ---
                    {relevent_sections}
                ---
            <|end|>
            <|user|>
                Question: {question_prompt}
            <|end|>
            <|assistant|>
                Answer:
    """


#new_ai_model = AiModel()
#new_ai_model.ask_a_question_from_pdf("./pdfs/2025-q1-earnings-transcript.pdf")