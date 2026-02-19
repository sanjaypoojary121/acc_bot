import logging
import sys
import json
import re
from flask import Flask, render_template, request, Response, jsonify
from core.engine import get_chat_engine

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = Flask(
    __name__,
    static_folder="frontend/static",
    template_folder="frontend/templates"
)

chat_engine = get_chat_engine()

VISUAL_TRIGGERS = [
    "diagram", "figure", "chart", "graph",
    "image", "picture", "show me", "draw"
]


def remove_ascii_diagrams(text):
    ascii_patterns = ["|", "+", "--", "->", "<-", "==="]
    for p in ascii_patterns:
        if p in text:
            return ""
    return text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    query_lower = query.lower()

    user_wants_visuals = any(trigger in query_lower for trigger in VISUAL_TRIGGERS)

    wants_explanation = any(word in query_lower for word in [
        "explain", "describe", "definition", "what is"
    ])

    wants_only_diagram = user_wants_visuals and not wants_explanation

    def generate():
        response = chat_engine.stream_chat(query)

        if not response.source_nodes:
            yield json.dumps({
                "type": "token",
                "content": "No data available in the provided documents."
            }) + "\n"
            return

        found_images = []
        for node in response.source_nodes:
            text = node.node.get_text()
            matches = re.findall(r"\[IMAGE_REF: (.*?)\]", text)
            found_images.extend(matches)

        found_images = list(set(found_images))

        full_response_text = ""

        for token in response.response_gen:

            clean_token = remove_ascii_diagrams(token)

            blocked_phrases = [
                "cannot provide the diagram",
                "copyright",
                "cannot create an image",
                "i can describe",
                "general overview",
            ]

            for phrase in blocked_phrases:
                if phrase in clean_token.lower():
                    clean_token = ""

            if wants_only_diagram:
                continue

            full_response_text += clean_token
            yield json.dumps({"type": "token", "content": clean_token}) + "\n"

        ai_references_visuals = (
            "figure" in full_response_text.lower()
            or "diagram" in full_response_text.lower()
        )

        if found_images and (user_wants_visuals or ai_references_visuals):
            html_content = "<br><br><b>⬇️ Visuals Found:</b><br>"
            for img_path in found_images:
                html_content += (
                    f'<img src="{img_path}" '
                    f'style="max-width:100%; border-radius:8px; '
                    f'border:1px solid #ccc; margin-top:10px;">'
                    f"<br>"
                )

            yield json.dumps({"type": "token", "content": html_content}) + "\n"

        sources = []
        for node in response.source_nodes:
            file_name = node.node.metadata.get("file_name", "Unknown")
            page_label = node.node.metadata.get("page_number", "N/A")
            sources.append({"file": file_name, "page": page_label})

        unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]
        yield json.dumps({"type": "sources", "content": unique_sources}) + "\n"

    return Response(generate(), mimetype="application/json")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
