import sqlite3
from Crypto.Cipher import AES
import os
import json

# Function to get the Chrome user data path
def get_chrome_user_data_path():
    if os.name == 'nt':  # Windows
        return os.path.expanduser(r"C:\Users\%s\AppData\Local\Google\Chrome\User Data" % os.getlogin())
    elif os.name == 'posix':  # macOS and Linux
        return os.path.expanduser(r"~/.config/google-chrome/Default")
    else:
        raise Exception("Unsupported OS")

# Function to get the encryption key from local state file
def get_encryption_key():
    with open(os.path.join(get_chrome_user_data_path(), "Local State"), "r", encoding="utf-8") as f:
        local_state = json.load(f)
    
    return base64.b64decode(local_state["os_crypt"]["encrypted_key"])[5:]

# Function to decrypt the password
def decrypt_password(encrypted_password, key):
    encrypted_password = base64.b64decode(encrypted_password)
    # Remove DPAPI header and take first 16 bytes as IV
    iv = encrypted_password[3:19]
    cipher = AES.new(key, AES.MODE_GCM, iv)
    return cipher.decrypt(encrypted_password[19:]).decode()

# Function to fetch passwords from Chrome database
def get_chrome_passwords():
    db_path = os.path.join(get_chrome_user_data_path(), "Login Data")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logins';")
    if not cursor.fetchone():
        print(f"Table 'logins' does not exist in {db_path}")
        return []
    
    cursor.execute("SELECT origin_url, action_url, username_value, password_value FROM logins")
    passwords = []
    encryption_key = get_encryption_key()
    
    for row in cursor.fetchall():
        url, _, username, encrypted_password = row
        if encrypted_password:
            password = decrypt_password(encrypted_password, encryption_key)
            passwords.append({"url": url, "username": username, "password": password})
    
    conn.close()
    return passwords

# Main function to print passwords
def main():
    passwords = get_chrome_passwords()
    for entry in passwords:
        print(f"URL: {entry['url']}")
        print(f"Username: {entry['username']}")
        print(f"Password: {entry['password']}\n")

if __name__ == "__main__":
    main()
