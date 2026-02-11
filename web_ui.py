import warnings
import logging
import git
import torch

# Try to import the Dictionary class in a way compatible with different fairseq versions
try:
    from fairseq.data.dictionary import Dictionary as FairseqDictionary
except ImportError:
    try:
        from fairseq.data import Dictionary as FairseqDictionary
    except ImportError:
        FairseqDictionary = None

if FairseqDictionary is not None:
    torch.serialization.add_safe_globals([FairseqDictionary])

warnings.filterwarnings("ignore")

# Set logging levels for noisy modules
for l in ["httpx", "uvicorn", "httpcore", "urllib3", "PIL", "faiss"]:
    logging.getLogger(l).setLevel(logging.ERROR)

# Now import shared after setting up logging
import shared
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn

from tabs.inference_tab import create_inference_tab
from tabs.train_tab import create_train_tab
from tabs.vocal_tab import create_vocal_tab
from tabs.ckpt_processing_tab import create_ckpt_processing_tab

# Create Gradio app
with gr.Blocks(title="RVC WebUI Fork") as gradio_app:
    try:
        repo = git.Repo(search_parent_directories=True)
        version_info = f"## RVC WebUI Fork ({repo.active_branch}) ({repo.head.object.hexsha[:7]})"
    except Exception:
        version_info = "## RVC WebUI Fork"
    
    gr.Markdown(version_info)
    
    with gr.Tabs():
        create_inference_tab(app=gradio_app)  # Pass gradio_app, not app
        create_vocal_tab()
        create_train_tab()
        create_ckpt_processing_tab()

# Handle launch based on environment
if shared.config.iscolab:
    # For Colab, launch directly with Gradio
    gradio_app.queue(max_size=1022).launch(share=True)
else:
    # For non-Colab, set up FastAPI with Gradio mounted
    gradio_app.queue(max_size=1022)  # Enable queuing
    
    # Create FastAPI app
    fastapi_app = FastAPI()
    
    # Add CORS middleware
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add private network header middleware
    @fastapi_app.middleware("http")
    async def add_private_network_header(request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response
    
    # Mount Gradio app
    gr.mount_gradio_app(fastapi_app, gradio_app, path="/gradio")
    
    # Redirect root to /gradio
    @fastapi_app.get("/")
    async def redirect_to_gradio():
        return RedirectResponse(url="/gradio")
    
    # Configure logging
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    
    print(f"Listening on http://0.0.0.0:{shared.config.listen_port}")
    
    # Run the server
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=shared.config.listen_port,
        log_level="warning",
        access_log=False,
    )
