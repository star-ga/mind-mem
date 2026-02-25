FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY scripts/ scripts/
COPY mcp_server.py ./

RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENTRYPOINT ["python3", "mcp_server.py"]
