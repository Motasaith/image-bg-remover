import pandas as pd
import google.generativeai as genai
import time
import os
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = "AIzaSyAgmHhd0Zly0xCh8KjLcMA37dwtIGRcLps" 
INPUT_FILE = "users.csv"
OUTPUT_FILE = "arranged_names.csv"
BATCH_SIZE = 20   

# Configure API
genai.configure(api_key=API_KEY)
MODEL_NAME = 'gemini-2.0-flash' 
model = genai.GenerativeModel(MODEL_NAME)

print(f"--- AI Name Cleaner (Rate Limit Safe) ---")
print(f"Using Model: {MODEL_NAME}")

def get_smart_names_batch(emails):
    """
    Sends a batch of emails to Gemini and asks for cleaned names.
    Includes auto-retry for Rate Limits.
    """
    email_list_str = "\n".join([str(e) for e in emails])
    
    prompt = f"""
    You are a data processing assistant. I will provide {len(emails)} email addresses.
    Extract a sensible "First Name" or "Full Name" for a user profile from each.
    
    Rules:
    1. Remove all numbers (e.g., 'mhamza5431' -> 'Mhamza').
    2. Split names that are stuck together using common logic (e.g., 'umairzaheer' -> 'Umair Zaheer', 'abdullahzafar' -> 'Abdullah Zafar').
    3. Expand 'm' prefix to 'Muhammad' ONLY if it makes sense (e.g., 'mhamza' -> 'Muhammad Hamza').
    4. Capitalize names properly (Title Case).
    5. If the email is random letters, numbers, or gibberish (e.g., '12345', 'x78y9'), return EXACTLY "Null".
    6. Return EXACTLY {len(emails)} lines. Do not add bullet points or intro text.
    
    Emails:
    {email_list_str}
    """
    
    # Retry loop
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            cleaned_names = response.text.strip().split('\n')
            cleaned_names = [name.replace('*', '').replace('-', '').strip() for name in cleaned_names]
            
            # Verify count
            if len(cleaned_names) != len(emails):
                while len(cleaned_names) < len(emails):
                    cleaned_names.append("Null")
                cleaned_names = cleaned_names[:len(emails)]
            
            return cleaned_names

        except Exception as e:
            error_msg = str(e)
            # Check specifically for Rate Limit (429) or Quota errors
            if "429" in error_msg or "Quota" in error_msg or "429" in error_msg:
                print(f"\n[!] Rate limit hit. Pausing for 60 seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(60) # Wait 1 minute
                continue # Try again
            else:
                print(f"\n[!] API Error: {e}")
                return ["Null"] * len(emails)
    
    print("\n[!] Failed after max retries.")
    return ["Null"] * len(emails)

def main():
    file_path = input(f"Enter CSV filename (press Enter for '{INPUT_FILE}'): ").strip() or INPUT_FILE
    
    if not os.path.exists(file_path):
        print("Error: File not found! Please put your CSV in this folder.")
        return

    try:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        print(f"Error reading file: {e}")
        return
            
    # Find email column
    email_col = None
    for col in df.columns:
        if "email" in col.lower():
            email_col = col
            break
    
    if not email_col:
        print("Error: Could not find an 'email' column.")
        return

    print(f"Processing {len(df)} rows... (This will take time to avoid errors)")
    
    emails = df[email_col].tolist()
    all_cleaned_names = []

    # Process
    for i in tqdm(range(0, len(emails), BATCH_SIZE)):
        batch = emails[i : i + BATCH_SIZE]
        batch_results = get_smart_names_batch(batch)
        all_cleaned_names.extend(batch_results)
        
        # SAFETY PAUSE: Wait 4 seconds between batches to stay under 15 requests/min
        time.sleep(4) 

    # Save
    df['Arranged_Name'] = all_cleaned_names
    if 'firstname' in df.columns:
         df['firstname'] = df['Arranged_Name']
    else:
         df['firstname'] = df['Arranged_Name']

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccess! Processed file saved as: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()