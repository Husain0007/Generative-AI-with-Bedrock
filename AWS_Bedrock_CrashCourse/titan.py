# Image generation with Titan
import boto3
import json
import base64
import os 

prompt_data = "provide me a 4K UHD image of a New York."

bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "textToImageParams": {
        "text": prompt_data
    },
    "taskType": "TEXT_IMAGE",
    "imageGenerationConfig": {
        "cfgScale": 10,
        "seed": 0,
        "quality": "standard",
        "width": 1024,
        "height": 1024,
        "numberOfImages": 1
    }
}

body = json.dumps(payload)
model_id = "amazon.titan-image-generator-v1"
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get("body").read())
print(response_body)
artifact = response_body.get("images")[0]  # Changed from "artifacts" to "images"
image_encoded = artifact.encode("utf-8")
image_bytes = base64.b64decode(image_encoded)  # Changed from b64encode to b64decode

# Save image to a file in the output directory.
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/generated-img.png"
with open(file_name, "wb") as f:
    f.write(image_bytes)