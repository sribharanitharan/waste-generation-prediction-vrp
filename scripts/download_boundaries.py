import urllib.request
import os
import json

out_dir = r"c:\clg\9thsem\projects\Spatial\data\raw"
os.makedirs(out_dir, exist_ok=True)

# 1. BBMP Bengaluru boundary
url_bbmp = "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Bangalore/BBMP.geojson"
bbmp_file = os.path.join(out_dir, "bengaluru_wards.geojson")

# 2. Karnataka District boundary
url_kar = "https://raw.githubusercontent.com/civictech-India/INDIA-GEO-JSON-Datasets/main/Karnataka_District_Boundary.json"
kar_file = os.path.join(out_dir, "karnataka_districts.geojson")

def download_file(url, target_path):
    print(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        content = response.read()
    with open(target_path, "wb") as f:
        f.write(content)
    # verify json
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Saved {target_path} successfully ({len(data.get('features', []))} features, {len(content):,} bytes)")

try:
    download_file(url_bbmp, bbmp_file)
except Exception as e:
    print("Error BBMP:", e)

try:
    download_file(url_kar, kar_file)
except Exception as e:
    print("Error Karnataka:", e)

print("Boundary download completed!")
