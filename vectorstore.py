import wikipediaapi
import subprocess
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import base64
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
from crewai_tools import tool, BaseTool
import logging
import warnings
from langchain_community.tools import DuckDuckGoSearchResults

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["USER_AGENT"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

duckduckgo_tool = DuckDuckGoSearchResults()


@tool("google search tool.")
def google_search_tool(argument: str) -> str:
    """Use this tool to search new information using google, if you do need more information."""
    return get_google_content(argument) # string to be sent back to the agent


@tool ("duckduck go search tool")
def duck_duck_go_search_tool(argument: str) -> str:
    """Use this tool to search new information by crawling the web, if you do need additional information but do not use json as input!"""
    print("Duck: ", argument)
    response = duckduckgo_tool.invoke(argument)
    return response

@tool("wikipedia search tool")
def wikipedia_search_tool(argument: str) -> str:
    """Use this tool to search a single keyword on wikipedia, if you do not know the answer to a question."""
    return get_wiki_content(argument)

@tool("calculator")
def calculate_tool(operation: str) -> str:
    """Use this tool to calculate result of a mathematical expression"""
    return eval(operation)

def get_google_content(text, max_results=3):
    socs_cookie = base64.encodebytes(f"\b\x01\x12\x1C\b\x01\x12\x12gws_{datetime.today().strftime('%Y%m%d')}-0_RC3\x1A\x02en \x01\x1A\x06\b\x80º¦±\x06".encode())
    headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/123.0.2420.97",
        "Cookie": f"SOCS={socs_cookie}",
    }

    url = 'https://google.com/search?q=' + text
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')
    for g in soup.find_all(class_='g'):
        print(g.text)
        print('-----')

    ## split then join to convert spaces to + in link
    url = 'https://google.com/search?q=' + '+'.join(text.split())
    print('From', url, '\n---\n')
    soup = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")

    ## loop through only the first results up to max_results
    for d in soup.select('div:has(>div>a[href] h3)')[:max_results]:
        print(d.h3.get_text(' ').strip()) ## title

        ## link
        res_link = d.select_one('a[href]:has(h3)').get('href')
        if res_link.startswith('/url?q='):
            res_link = res_link.split('=',1)[1].split('&')[0]
        print(res_link)

        ## description
        sel_typ = d.select_one('div:has(>a[href] h3)+div')
        if sel_typ != None:
            print(sel_typ.get_text(' ').strip())

        print('\n---\n') ## separate results

def get_wiki_content(key_word):
    try:
        #  Wikipedia API ready
        wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
        page = wiki_wiki.page(key_word)
        if page.exists(): # Page - Exists: True
            print("Page - Title:", page.title)
            print("Page - Summary:", page.summary)
            return page.summary
        else:
            print("Page not found.")
        return page.summary
    except Exception as error:
        # handle the exception
        print("An exception occurred:", error)
    return ""

class ReadFile(BaseTool):
    name: str = "ReadFile"
    description: str = "Only use this tool if you want to read a file content on this computer. Only execute this tool if you want to read a file!"

    def _run(self, filename: str):
        try:
            if len(filename) != 0:
                if filename != "":
                    print(filename)
                print("Filename found")
                try:
                    # Using subprocess to execute 'cat' command
                    result = subprocess.run(['cat', filename],  capture_output=True,  text=True,  check=True)
                    print(f"Reading file: {filename}")
                    print(result.stdout)
                    return result.stdout
                except subprocess.CalledProcessError as e:
                    error_msg = f"Error reading file with cat: {str(e)}"
                    print(error_msg)
                    return error_msg
                except Exception as error:
                    error_msg = f"Error: {str(error)}"
                    print(error_msg)
                    return error_msg
        except Exception:
            return "This is not a valid python code search syntax. Try a different string based syntax."

        def _arun(self, radius: int):
            raise NotImplementedError("This tool does not support async")
read_file_tool = ReadFile()


from llmsherpa.readers import LayoutPDFReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from typing import Any
import os
from glob import glob
from pydantic import Field
warnings.filterwarnings("ignore", message="Support for class-based `config` is deprecated, use ConfigDict instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.11/migration/", category=DeprecationWarning)

class PDFVectorStoreTool(BaseTool):
    name: str = "pdf_vector_store"
    description: str = "Use this tool to read AI ECG classification related PDFs to gain a deep ecg classificaiton knowledge and understanding."
    # Define class attributes properly with Field
    # Online pdf converter
    #llmsherpa_api_url: str = Field(default="https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all")
    # Local pdf converter
    llmsherpa_api_url: str = Field(default="http://localhost:5001/api/parseDocument?renderFormat=all")
    pdf_reader: Any = Field(default=None)
    text_splitter: Any = Field(default=None)
    embeddings: Any = Field(default=None)
    vector_store: Any = Field(default=None)
    dir: str = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize components
        self.pdf_reader = LayoutPDFReader(self.llmsherpa_api_url)
        self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # About 1500-2000 tokens, safe for 8k context
        chunk_overlap=200,
        length_function=len,
            separators=[
            "\n\n\n",  # Triple line breaks (paragraphs)
            "\n\n",    # Double line breaks (paragraphs)
            "\n",      # Single line breaks
            # ". ",      # Sentences
            # "? ",      # Questions
            # "! ",      # Exclamations
        ],
        keep_separator=True,
        is_separator_regex=False
        )

        # Set logging level to ERROR to suppress warnings
        logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

        #Initialize HuggingFace embeddings with a good model for semantic search
        self.embeddings = HuggingFaceEmbeddings(
            # Online embeddings
            model_name="sentence-transformers/all-mpnet-base-v2",
            # Local embddings
            #model_name="./all-mpnet-base-v2",
            model_kwargs={'device': 'cuda'}, # Use GPU if available
            encode_kwargs={'normalize_embeddings': True} # Normalize embeddings for better similarity search
        )
        self.vector_store = None
        self.dir = "./books/lorem_ipsum"
        self.create_vector_store(self.dir)

    def vectorstore_get_num_items(self):
        return self.vector_store.index.ntotal

    def vectorstore_search_id_by_contex(self, context):
        for _id, doc in self.vector_store.docstore._dict.items():
            if(context in doc.page_content):
                return _id
        return 0

    def vectorstore_delete_by_contex(self, context):
        for _id, doc in self.vector_store.docstore._dict.items():
            if(context in doc.page_content):
                id_id = []

    def vectorstore_get_num_items(self):
        return self.vector_store.index.ntotal

    def vectorstore_search_id_by_contex(self, context):
        for _id, doc in self.vector_store.docstore._dict.items():
            if(context in doc.page_content):
                return _id
        return 0

    def vectorstore_delete_by_contex(self, context):
        for _id, doc in self.vector_store.docstore._dict.items():
            if(context in doc.page_content):
                id_id = []
                id_id.append(_id)
                self.vector_store.delete(id_id[0:1])
        return 0

    def vectorstore_clear(self):
        for k in range(self.vector_store.index.ntotal):
            copy_safe = self.vector_store.docstore._dict.items()
            for _id, doc in copy_safe:
                id_id = []
                id_id.append(_id)
                self.vector_store.delete(id_id[0:1])
                break
        return 0

    def vectorstore_delete_by_id(self, id):
        id_id = []
        id_id.append(id)
        self.vector_store.delete(id_id[0:1])

    def vectorstore_dump(self):
        k=0
        for _id, doc in self.vector_store.docstore._dict.items():
            print(k, _id, doc)
            print(self.vector_store.index_to_docstore_id.items())
            print(self.vector_store.index_to_docstore_id.values())
            print(self.vector_store.index_to_docstore_id.get(0))
            k += 1

    def save_vector_store(self):
        self.vector_store.save_local(self.dir + "/vectorstore/vectorstore.db")

    def load_vector_store(self):
        try:
            self.vector_store = FAISS.load_local(self.dir + "/vectorstore/vectorstore.db", self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print("Error: Loading vectorstore failed:", e)


    def search_vectorstore(self, input_query: str):
        results = self.vector_store.similarity_search_with_score(query=input_query,k=1)
        for doc, score in results:
            print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]" + "\n---\n")

    def vectorstore_search(self, query: str, k: int = 2) -> str:
        """Search the vector store for relevant content"""
        try:
            # Try to load vector store if not exists
            if not self.vector_store:
                return "No vector store exists. Please create one first."

            results = self.vector_store.similarity_search(query, k=k)
            formatted_results = []
            for doc in results:
                source = doc.metadata.get("source", "unknown source")
                content = doc.page_content
                formatted_results.append(f"From {source}:\n{content}\n")

            ret = "\n---\n".join(formatted_results)
            print(ret)
            return ret

        except Exception as e:
            return f"Error searching vector store: {str(e)}"


    def create_vector_store(self, directory_path: str) -> str:
        """Create a FAISS vector store from all PDFs in a directory"""
        try:
            # Save vector store
            if not os.path.exists(self.dir + "/vectorstore"):
                # Verify input directory exists
                if not os.path.exists(directory_path):
                    return f"Error: Directory not found at {directory_path}"

                # Get all PDF files in directory
                pdf_files = glob(os.path.join(directory_path, "*.pdf"))
                if not pdf_files:
                    return f"No PDF files found in {directory_path}"

                print(f"Processing {len(pdf_files)} PDF files from {directory_path}")

                all_documents = []
                for pdf_path in pdf_files:
                    try:
                        print(f"Processing {pdf_path}")
                        # Read PDF
                        doc = self.pdf_reader.read_pdf(pdf_path)

                        # Extract text from all pages
                        text_content = []
                        for chunk in doc.chunks():
                            text_content.append(chunk.to_context_text())

                        # Create documents
                        full_content = "\n".join(text_content)
                        texts = self.text_splitter.split_text(full_content)
                        documents = [Document(page_content=t, metadata={"source": pdf_path}) for t in texts]
                        all_documents.extend(documents)


                    except Exception as e:
                        print(f"Error processing {pdf_path}: {str(e)}")
                        continue

                if not all_documents:
                    return "No documents were successfully processed"
                # Create vector store
                self.vector_store = FAISS.from_documents(all_documents, self.embeddings)
                print("Directory not found")
                os.makedirs(self.dir + "/vectorstore", exist_ok=True)
                self.save_vector_store()
            else:
                self.load_vector_store()
                # Test vectorstore similarity search
                #self.search_vectorstore("lorem ipsum")

            print(f"Vector store created successfully from PDFs and saved.")

        except Exception as e:
            print(f"Error creating vector store: {str(e)}")


    def _run(self, command: str) -> str:
        """Run the tool with query command"""
        try:
            return self.vectorstore_search(command)
        except Exception as e:
            return f"Error processing command: {str(e)}"

    def _arun(self, command: str) -> str:
        """Async implementation"""
        raise NotImplementedError("This tool does not support async")
