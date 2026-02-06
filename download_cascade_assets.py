import os
import gdown

# --- Set the folder where you want to save the assets ---
project_dir = r"C:\Dev\cascade"  # change to your MCP server folder
assets_dir = os.path.join(project_dir, "cascade_assets")
os.makedirs(assets_dir, exist_ok=True)

# --- Google Drive folder ID (from GREmLN Quickstart) ---
drive_folder_url = "https://drive.google.com/drive/folders/1cMR9HoAC22i6sKSWgfQUEQRf0UP_w3_m"

# --- Download everything from the folder ---
print(f"Downloading GREmLN assets to {assets_dir} ...")
gdown.download_folder(drive_folder_url, output=assets_dir, use_cookies=False)

print("Download complete!")
print(f"Check {assets_dir} for model.ckpt, networks/, and demo data.")
