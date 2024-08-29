from pyngrok import ngrok
import os

# Start ngrok to tunnel the Streamlit port
port = 8505
public_url = ngrok.connect(port)
print(f"Public URL: {public_url}")

# Run the Streamlit app
os.system(f"streamlit run app.py --server.port {port}")
