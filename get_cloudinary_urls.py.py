import cloudinary
import cloudinary.api
import pandas as pd

# ----------------------------
# Configure Cloudinary
# ----------------------------
cloudinary.config(
    cloud_name="yrb1ur3i",
    api_key="528527643576419",
    api_secret="f1O_Vm0NWNXdyc-5Gds3FeRrTqI",
    secure=True
)

folders = [
    "brain_mri/healthy",
    "brain_mri/glioma",
    "brain_mri/meningioma",
    "brain_mri/pituitary"
]

rows = []

for folder in folders:
    print(f"Reading {folder}...")
    result = cloudinary.api.resources(
        type="upload",
        prefix=folder,
        max_results=500
    )

    for img in result["resources"]:
        rows.append({
            "Category": folder.split("/")[-1],
            "PublicID": img["public_id"],
            "URL": img["secure_url"]
        })

df = pd.DataFrame(rows)
df.to_csv("cloudinary_images.csv", index=False)

print("--------------------------------")
print(df.head())
print("--------------------------------")
print(f"Total Images Found: {len(df)}")
print("CSV file saved as cloudinary_images.csv")
