import json
import os

bbmp_src = r"C:\Users\DELL\.gemini\antigravity-ide\brain\656d077d-ed09-4bcd-acb6-ac10b7eba0fc\.system_generated\steps\145\content.md"
kar_src  = r"C:\Users\DELL\.gemini\antigravity-ide\brain\656d077d-ed09-4bcd-acb6-ac10b7eba0fc\.system_generated\steps\153\content.md"

out_dir = r"c:\clg\9thsem\projects\Spatial\data\raw"
os.makedirs(out_dir, exist_ok=True)

def extract_geojson(src_path, dst_path):
    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Find start of json '{'
    idx = text.find("{\n\"type\"")
    if idx == -1:
        idx = text.find('{"type"')
    if idx == -1:
        idx = text.find('{')
    json_text = text[idx:].strip()
    # Verify valid json
    data = json.loads(json_text)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Saved: {dst_path} with {len(data.get('features', []))} features.")

extract_geojson(bbmp_src, os.path.join(out_dir, "bengaluru_wards.geojson"))
extract_geojson(kar_src, os.path.join(out_dir, "karnataka_districts.geojson"))
print("Done!")
