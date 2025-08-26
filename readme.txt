pip install -r requirements.txt
# set env in .env (OpenAI or Azure OpenAI)

# 1) Index
python hier_rag_langchain.py --build ./data

# 2) Ask
python hier_rag_langchain.py --ask "What does the spec say about token limits and rate limiting?"
