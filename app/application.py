from flask import Flask, render_template, request, session, redirect, url_for
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

app = Flask(__name__)
app.secret_key = os.urandom(24)

from markupsafe import Markup
def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br


@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("prompt")
        print("✅ CHECKPOINT 1: Received user input ->", user_input)

        if user_input:
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages
            print("✅ CHECKPOINT 2: Stored user input in session")

            try:
                # Create QA chain
                print("✅ CHECKPOINT 3: Creating QA chain...")
                qa_chain = create_qa_chain()
                
                if qa_chain is None:
                    raise Exception("QA chain could not be created (LLM or VectorStore issue)")
                print("✅ CHECKPOINT 4: QA chain created successfully")

                # Run query
                response = qa_chain.invoke({"query": user_input})
                print("✅ CHECKPOINT 5: QA chain response ->", response)

                # Extract result
                result = response.get("result", "No response")
                print("✅ CHECKPOINT 6: Extracted result ->", result)

                messages.append({"role": "assistant", "content": result})
                session["messages"] = messages
                print("✅ CHECKPOINT 7: Stored assistant response in session")

            except Exception as e:
                import traceback
                print("❌ ERROR:", str(e))
                traceback.print_exc()

        return redirect(url_for("index"))

    print("✅ CHECKPOINT 8: Rendering chat UI with messages ->", session.get("messages", []))
    return render_template("index.html", messages=session.get("messages", []))


@app.route("/clear")
def clear():
    print("✅ CHECKPOINT 9: Clearing session messages")
    session.pop("messages", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    print("✅ CHECKPOINT 0: Starting Flask app on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
