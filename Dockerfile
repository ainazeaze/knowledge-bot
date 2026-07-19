FROM python:3.12-slim

WORKDIR /app

# Install CPU-only torch before the rest to keep the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ChromaDB persists here — mount a volume to keep data across restarts
VOLUME ["/app/chroma_data"]

CMD ["python", "interfaces/bot.py"]
