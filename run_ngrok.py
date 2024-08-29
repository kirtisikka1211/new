from pyngrok import ngrok, conf
import os

# Set ngrok auth token
ngrok.set_auth_token("2lIKl9q7oYG4hMrktMGvB7gAlBL_2YCXjHuzv5cfZapkRwxu7")

# Start ngrok to tunnel the Streamlit port
port = 8000
public_url = ngrok.connect(port)
print(f"Public URL: {public_url}")

# Run the Streamlit app
os.system(f"streamlit run app.py --server.port {port}")
