import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fpdf import FPDF
from datetime import datetime
from ultralytics import YOLO
import os


def do_inference(photo_path: str):
    """
    Roda YOLO e salva imagem anotada.
    """
    model = YOLO("best.pt")
    results = model.predict(source=photo_path, conf=0.50)
    result = results[0]

    image = Image.open(photo_path).convert("RGB")
    original_np = np.array(image)

    annotated = result.plot()

    if annotated is not None:
        final_image = annotated[..., ::-1]  # BGR -> RGB
    else:
        final_image = original_np

    outname = f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    plt.figure(figsize=(10, 5))
    plt.imshow(final_image)
    plt.axis("off")
    plt.savefig(outname)
    plt.close()

    return outname


def gerar_pdf(perguntas_respostas, image_path, pdf_path="results/resultado.pdf"):
    """
    perguntas_respostas → lista de tuplas (pergunta, resposta)
    """

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Fonte
    font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
    pdf.add_font("AppFont", "", font_path, uni=True)
    pdf.set_font("AppFont", "", 14)

    # Título
    pdf.set_font("AppFont", "", 20)
    pdf.set_text_color(0, 70, 140)
    pdf.cell(0, 12, "Relatório de Respostas", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("AppFont", "", 12)
    pdf.set_text_color(50, 50, 50)

    # Blocos de perguntas e respostas
    for pergunta, resposta in perguntas_respostas:
        pdf.set_font("AppFont", "", 14)
        pdf.set_text_color(0, 70, 140)
        pdf.cell(0, 8, f"• {pergunta}", ln=True)

        pdf.set_font("AppFont", "", 12)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 7, resposta)
        pdf.ln(3)

    # Imagem
    pdf.ln(5)
    pdf.set_font("AppFont", "", 14)
    pdf.set_text_color(0, 70, 140)
    pdf.cell(0, 10, "Imagem analisada:", ln=True)
    pdf.ln(5)

    try:
        img = Image.open(image_path)
        w, h = img.size

        max_w = 170
        max_h = 140
        scale = min(max_w / w, max_h / h)

        new_w = w * scale
        new_h = h * scale

        x = (210 - new_w) / 2
        y = pdf.get_y()

        pdf.image(image_path, x=x, y=y, w=new_w, h=new_h)
        pdf.set_y(y + new_h + 10)

    except Exception as e:
        pdf.set_text_color(255, 0, 0)
        pdf.multi_cell(0, 8, f"Erro ao carregar imagem: {e}")

    # SEMPRE gerar o PDF aqui
    pdf.output(pdf_path)
