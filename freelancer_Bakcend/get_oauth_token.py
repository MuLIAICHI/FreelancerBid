import requests
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Your Freelancer app credentials from the console
CLIENT_ID = "df8a0056-17e3-413f-bbb4-c063ca2de99c"
CLIENT_SECRET = "b50f1e3f5d28fc7c47f4ec75a56d08fa6c1f4c4efa2a49ee911b106cc99d1b77730abd3c87aa677a080fb8b47dfca81c14e82d6567609a83e1943aa4f3d4f5d0"
REDIRECT_URI = "http://localhost:8000/callback"

# Use the production environment URLs
AUTH_URL = "https://accounts.freelancer.com/oauth/authorize"
TOKEN_URL = "https://accounts.freelancer.com/oauth/token"

# Step 1: Redirect user to authorization page
auth_url = f"{AUTH_URL}?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=basic+fln:project_manage"
print(f"Opening browser to authorize your app: {auth_url}")
webbrowser.open(auth_url)

# Step 2: Create a simple server to receive the callback
class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        
        if 'code' in params:
            # Extract authorization code
            auth_code = params['code'][0]
            print(f"Received authorization code: {auth_code}")
            
            # Exchange code for access token
            data = {
                'grant_type': 'authorization_code',
                'code': auth_code,
                'client_id': CLIENT_ID,
                'client_secret': CLIENT_SECRET,
                'redirect_uri': REDIRECT_URI
            }
            
            print(f"Exchanging code for token at {TOKEN_URL}...")
            response = requests.post(TOKEN_URL, data=data)
            
            if response.status_code == 200:
                token_data = response.json()
                
                # Print the access token (this is your OAuth token)
                print("\n=== YOUR OAUTH TOKEN ===")
                print(token_data['access_token'])
                print("========================\n")
                
                # Also print refresh token if available
                if 'refresh_token' in token_data:
                    print("Refresh token:", token_data['refresh_token'])
                    print("Token expires in:", token_data.get('expires_in', 'unknown'), "seconds")
                
                # Create .env file with the token
                with open('.env', 'w') as f:
                    f.write(f"FLN_URL=https://www.freelancer.com\n")
                    f.write(f"FLN_OAUTH_TOKEN={token_data['access_token']}\n")
                print("Saved token to .env file!")
                
                # Send success response to browser
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"Authentication successful! You can close this window and check your console for the token.")
            else:
                print(f"Error exchanging code for token: {response.status_code}")
                print(response.text)
                
                self.send_response(500)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"Error exchanging code for token. Check console for details.")
        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Authentication failed! No authorization code received.")

# Start the server to receive callback
print("Starting local server on port 8000...")
server = HTTPServer(('localhost', 8000), CallbackHandler)
print("Waiting for authentication callback...")
server.handle_request()  # Handle one request then close