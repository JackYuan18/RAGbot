from flask import Flask, render_template, request, jsonify
import sys
from TooyRag import rag_application as rag_query
# Placeholder for your RAG application
# Assume you have a function like this in your RAG code:
# def rag_query(user_message):
#     # Your RAG logic here: retrieve documents, generate response using LLM, etc.
#     return "RAG response to: " + user_message

# If your RAG is in a separate file, import it here.
# For example: from your_rag_module import rag_query

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if user_message:
        # Run your RAG application here
        response,answer = rag_query(user_message)  # Replace with your actual RAG function
        return jsonify({'response': 'hi'})
    return jsonify({'response': "Sorry, no message received."})

if __name__ == '__main__':
    # app.run(debug=True)
    user_message="What does human driving data provide?"
    response,answer = rag_query(user_message)
    print(answer)