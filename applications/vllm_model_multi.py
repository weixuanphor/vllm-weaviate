import os
import psutil
import requests
import streamlit as st
import subprocess
import signal
import time
import weaviate

# -------------------------------
# Constants & Utilities
# -------------------------------
MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.2-3B-Instruct"
]
API_PORT = 8000
API_URL = f"http://localhost:{API_PORT}/v1/completions"
TOP_K = 2

class VLLMServerManager:
    def __init__(self, port=API_PORT):
        self.port = port
        self.process = None
        self.current_model = None
        self.log_file = f"logs/vllm_multi_{self.port}.log"
        
        if not os.path.exists("logs"):
            os.makedirs("logs")

    def start_server(self, model_name):
        self.stop_server()  # kill old first

        cmd = [
            "vllm", "serve", model_name,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--tensor-parallel-size", "4",
            "--max-model-len", "32768",
            "--enforce-eager"
        ]
        log_f = open(self.log_file, "w")
        self.process = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
        log_f.close()
        self.current_model = model_name

        self._wait_for_ready()

    def _wait_for_ready(self, timeout=300):
        start = time.time()
        while time.time() - start < timeout:
            if not os.path.exists(self.log_file):
                time.sleep(1)
                continue
            with open(self.log_file, "r") as f:
                logs = f.read()
            if "Application startup complete." in logs:
                return
            if self.process.poll() is not None:
                raise RuntimeError("vLLM server exited unexpectedly")
            time.sleep(1)
        raise TimeoutError("vLLM server did not become ready in time")

    def stop_server(self, notify_fn=None):
        if self.process and self.process.poll() is None:
            if notify_fn:
                notify_fn("Stopping previous model server... Please wait.")
            self._kill_process_tree(self.process.pid)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if notify_fn:
                    notify_fn("Process did not terminate in time, force killing...")
                self._kill_process_tree(self.process.pid)
            self.process = None
            self.current_model = None   
    
    def _kill_process_tree(self, pid):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for p in children:
                p.terminate()
            gone, alive = psutil.wait_procs(children, timeout=5)
            for p in alive:
                p.kill()
            parent.terminate()
            parent.wait(5)
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            print(f"Error killing process tree: {e}")

def generate_text(model=None, query_text="", context="", temperature=0.7, top_p=0.9):
    prompt = f"""You are an expert assistant. Based on the following documents, answer the question. (Note: If the documents are irrelevant, ignore mentioning them in the answer.)

    Documents:
    {context}

    Question:
    {query_text}

    Answer:"""

    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    try:
        response = requests.post(API_URL, json=data, timeout=15)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]
    except Exception as e:
        st.error(f"Request failed: {e}")
        return ""


# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")

with st.spinner("Connecting to Weaviate..."):
    while True:
        try:
            if client.is_ready():
                break
        except Exception:
            pass
        time.sleep(1)

# Page Selection
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📚 Document Viewer", "💬 Chat With Model"])

# -------------------------------
# Page 1: Document Viewer
# -------------------------------
if page == "📚 Document Viewer":
    st.title("Document Viewer & Manager")

    def get_document_list():
        schema = client.schema.get()
        return [cls["class"] for cls in schema.get("classes", [])]

    def create_class_if_missing(class_name):
        classes = [c["class"].lower() for c in client.schema.get().get("classes", [])]
        if class_name.lower() in classes:
            st.info(f"Class '{class_name}' already exists.")
            return False
        try:
            client.schema.create_class({
                "class": class_name,
                "vectorizer": "text2vec-transformers",
                "properties": [
                    {"name": "title", "dataType": ["string"]},
                    {"name": "content", "dataType": ["text"]}
                ]
            })
            st.success(f"Created new class '{class_name}'.")
            return True
        except UnexpectedStatusCodeException as e:
            if "already exists" in str(e).lower():
                st.warning(f"Class '{class_name}' already exists (caught).")
                return False
            else:
                raise e

    def upload_documents(class_name, docs):
        existing_titles = set()
        try:
            res = client.query.get(class_name, ["title"]).do()
            objs = res.get("data", {}).get("Get", {}).get(class_name, [])
            existing_titles = {obj["title"] for obj in objs}
        except Exception as e:
            st.error(f"Failed to fetch existing documents in '{class_name}': {e}")
            return

        skipped, uploaded = 0, 0
        for doc in docs:
            if doc["title"] in existing_titles:
                st.warning(f"Skipped: `{doc['title']}` already exists in `{class_name}`.")
                skipped += 1
            else:
                try:
                    client.data_object.create(doc, class_name)
                    uploaded += 1
                except Exception as e:
                    st.error(f"Failed to upload `{doc['title']}`: {e}")
        
        if uploaded:
            st.success(f"Uploaded {uploaded} new document(s) to '{class_name}'.")
        if skipped:
            st.info(f"Skipped {skipped} duplicate document(s).")

    # Sidebar: Upload documents
    st.sidebar.header("Upload Documents")

    # Reset form fields on rerun
    if st.session_state.get("reset_fields"):
        st.session_state.new_class_input = ""
        st.session_state.existing_class_select = ""
        st.session_state.reset_fields = False
        
    new_class_name = st.sidebar.text_input(
        "New class name (leave blank to add to existing)",
        key="new_class_input"
    )
    current_docs = get_document_list()
    existing_class = None
    if current_docs:
        existing_class = st.sidebar.selectbox(
            "Or select existing class:",
            [""] + current_docs,
            key="existing_class_select"
        )
    else:
        st.sidebar.info("No existing classes found.")

    uploaded_files = st.sidebar.file_uploader("Upload TXT files", type=["txt"], accept_multiple_files=True)
    upload_trigger = st.sidebar.button("Upload")

    target_class = None
    if new_class_name.strip():
        target_class = new_class_name.strip()
    elif existing_class:
        target_class = existing_class

    if upload_trigger:
        if not uploaded_files:
            st.sidebar.warning("No files uploaded.")
        elif not new_class_name.strip() and not existing_class:
            st.sidebar.warning("Specify a new class name or select an existing class.")
        else:
            docs = []
            for f in uploaded_files:
                try:
                    content = f.read().decode("utf-8")
                    docs.append({"title": f.name, "content": content})
                except Exception as e:
                    st.sidebar.error(f"Error reading {f.name}: {e}")

            if docs:
                if new_class_name.strip():
                    created = create_class_if_missing(new_class_name.strip())
                    if created:
                        upload_documents(new_class_name.strip(), docs)
                        st.session_state.upload_success_msg = f"Uploaded to '{new_class_name.strip()}' successfully."
                        st.session_state.reset_fields = True
                elif existing_class:
                    upload_documents(existing_class, docs)
                    st.session_state.upload_success_msg = f"Uploaded to '{existing_class}' successfully."
                    st.session_state.reset_fields = True

    st.subheader("📂 Existing Document Classes")

    # Show documents in classes
    if current_docs:
        for class_name in current_docs:
            with st.expander(class_name):
                try:
                    res = client.query.get(class_name, ["title", "content"]).do()
                    docs = res.get("data", {}).get("Get", {}).get(class_name, [])
                    if docs:
                        for d in docs:
                            st.markdown(f"**📄 {d.get('title', 'Untitled')}**")
                            st.code(d.get("content", ""), language="text")
                    else:
                        st.write("No documents in this class.")
                except Exception as e:
                    st.error(f"Error reading '{class_name}': {e}")
    else:
        st.write("🕳️ No documents available.")

    
    if current_docs:
        docs_to_delete = st.multiselect("Select classes to delete:", current_docs, key="delete_select")
        if st.button("🗑️ Delete Selected Classes"):
            if docs_to_delete:
                for doc in docs_to_delete:
                    try:
                        client.schema.delete_class(doc)
                        st.success(f"Deleted class: `{doc}`")
                    except Exception as e:
                        st.error(f"Failed to delete `{doc}`: {e}")
                st.rerun()
            else:
                st.warning("No classes selected.")
    else:
        st.info("No document classes found.")

    st.write("---")

    # Delete individual text files from classes
    st.subheader("🗑️ Delete Text Files from a Document Class")
    if current_docs:
        selected_class = st.selectbox("Select class to manage files:", current_docs, key="file_delete_class")
        if selected_class:
            try:
                res = client.query.get(selected_class, ["title", "_additional { id }"]).do()
                objs = res.get("data", {}).get("Get", {}).get(selected_class, [])
                if objs:
                    options = [f"{obj['title']} ({obj['_additional']['id']})" for obj in objs]
                    selected_files = st.multiselect("Select files to delete:", options, key="files_to_delete")
                    if st.button(f"Delete selected files from `{selected_class}`", key="delete_files_button"):
                        if selected_files:
                            for fstr in selected_files:
                                obj_id = fstr.split("(")[-1].strip(")")
                                try:
                                    client.data_object.delete(obj_id, selected_class)
                                    st.success(f"Deleted file with ID {obj_id} from `{selected_class}`")
                                except Exception as e:
                                    st.error(f"Failed to delete file {fstr}: {e}")
                            st.rerun()
                        else:
                            st.warning("No files selected.")
                else:
                    st.info(f"No files in class `{selected_class}`.")
            except Exception as e:
                st.error(f"Failed to fetch objects from `{selected_class}`: {e}")
    else:
        st.info("No document classes found.")


# -------------------------------
# Page 2: Chat With Model
# -------------------------------
elif page == "💬 Chat With Model":
    st.title("vLLM Model Chat")

    if "server_manager" not in st.session_state:
        st.session_state.server_manager = VLLMServerManager()
        st.session_state.current_model = None
        st.session_state.output = ""

    model = st.selectbox("Choose a model (select to start server):", ["-- Select model --"] + MODELS)

    if model != "-- Select model --" and model != st.session_state.current_model:
        with st.spinner("Stopping previous model server... Please wait."):
            if st.session_state.current_model:
                st.session_state.server_manager.stop_server()
        with st.spinner(f"Starting server for {model}..."):
            st.session_state.server_manager.start_server(model)
        st.session_state.current_model = model
        st.session_state.output = ""

    question = st.text_area("Enter your question here:", height=150)

    if st.button("Generate") and question.strip():
        if st.session_state.current_model is None:
            st.warning("Please select a model first!")
        else:
            with st.spinner("Generating..."):
                try:
                    all_classes = client.schema.get().get("classes", [])
                    if not all_classes:
                        st.warning("No document classes available for context.")
                        context = ""
                    else:
                        # Select latest class
                        latest_class = all_classes[-1]["class"]
                        context = client.query.get(latest_class, ["title", "content"]) \
                            .with_near_text({"concepts": [question]}) \
                            .with_limit(TOP_K) \
                            .do()
                        if not context:
                            st.warning("No context retrieved.")
                    output = generate_text(model=st.session_state.current_model, query_text=question, context=context)
                    st.session_state.output = output
                except Exception as e:
                    st.error(f"Error fetching context or generating response: {e}")

    if st.session_state.output:
        st.subheader("Response:")
        st.write(st.session_state.output)
