import os
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from core.engine import get_chat_engine
import json

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")

print("🧠 Loading AI Engine...")
chat_engine = get_chat_engine()
print("✅ Ready for chat!")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "").strip()
    
    # Use stream_chat instead of chat
    streaming_response = chat_engine.stream_chat(user_query)

    # We create a generator function to yield data to the frontend
    def generate():
        # 1. Yield the text tokens as they come in
        for token in streaming_response.response_gen:
            yield json.dumps({"type": "token", "content": token}) + "\n"
        
        # 2. Once text is done, yield the sources
        sources = []
        if streaming_response.source_nodes:
            for node in streaming_response.source_nodes:
                meta = node.node.metadata
                sources.append({
                    "file": meta.get("file_name", "Unknown"),
                    "page": meta.get("page_label", "N/A")
                })
        yield json.dumps({"type": "sources", "content": sources}) + "\n"

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)