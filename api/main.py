from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from datetime import datetime
import os

from _utils import do_inference, gerar_pdf

#pip install fastapi uvicorn python-multipart ultralytics pillow matplotlib fpdf2 numpy


UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/uploadPhoto")
async def upload_photo(
    file: UploadFile = File(...),
    perguntas: List[str] = Form(...),
    respostas: List[str] = Form(...)
):
    """
    Recebe:
    - file : imagem
    - perguntas : lista de strings (Form)
    - respostas : lista de strings (Form)
    """

    # valida formato da imagem
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="O arquivo deve ser JPEG ou PNG"
        )

    # salva a imagem
    photo_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(photo_path, "wb") as f:
        f.write(await file.read())

    # roda YOLO
    result_image_path = do_inference(photo_path)

    # monta pares
    pares = list(zip(perguntas, respostas))

    # gera PDF final
    pdf_path = os.path.join(RESULTS_DIR, f"resultado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    gerar_pdf(pares, result_image_path, pdf_path)

    # retorna o PDF
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="resultado.pdf"
    )

#uvicorn main:app --reload --port 8080  
#^^^faz questao de rodar isso no terminal, na pasta certa