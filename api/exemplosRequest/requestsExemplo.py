import requests

url = "http://localhost:8080/uploadPhoto"

files = {
    "file": ("imagem.jpg", open("imagem.jpg", "rb"), "image/jpeg")
}

data = [ #fastAPI precisa que essas listas sejam enviadas desta 
    ("perguntas", "Nome"),
    ("perguntas", "Endereço"),
    ("perguntas", "Tipo de Construção"),
    ("perguntas", "Tipo de Material da Estrutura"),
    ("perguntas", "Data da aparição de falha"),
    ("perguntas", "Tamanho Aprox. da Falha (cm)"),
    ("perguntas", "Observações Adicionais"),
    ("respostas", "João Silva"),
    ("respostas", "Rua das Flores, 123"),
    ("respostas", "Ponte"),
    ("respostas", "Concreto"),
    ("respostas", "15/03/2024"),
    ("respostas", "50"),
    ("respostas", "Nenhuma")
]

resp = requests.post(url, files=files, data=data)

with open("resultado.pdf", "wb") as f:
    f.write(resp.content)
