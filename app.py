import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from diffusers import StableDiffusionInpaintPipeline
import os
import gc

# ------------------------------ CycleGAN Generator ------------------------------
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 3, 1, 1),
            torch.nn.InstanceNorm2d(channels),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(channels, channels, 3, 1, 1),
            torch.nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residuals=6):
        super().__init__()
        model = [
            torch.nn.Conv2d(in_channels, 64, 7, 1, 3),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(True),
        ]
        in_features = 64
        for _ in range(2):
            model += [
                torch.nn.Conv2d(in_features, in_features * 2, 3, 2, 1),
                torch.nn.InstanceNorm2d(in_features * 2),
                torch.nn.ReLU(True)
            ]
            in_features *= 2
        for _ in range(n_residuals):
            model += [ResidualBlock(in_features)]
        for _ in range(2):
            model += [
                torch.nn.ConvTranspose2d(in_features, in_features // 2, 3, 2, 1, output_padding=1),
                torch.nn.InstanceNorm2d(in_features // 2),
                torch.nn.ReLU(True)
            ]
            in_features = in_features // 2
        model += [torch.nn.Conv2d(64, out_channels, 7, 1, 3), torch.nn.Tanh()]
        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# ------------------------------ Pix2Pix Generator ------------------------------
class UNetGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def down_block(in_channels, out_channels, normalize=True):
            layers = [torch.nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
            if normalize:
                layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.LeakyReLU(0.2))
            return torch.nn.Sequential(*layers)

        def up_block(in_channels, out_channels, dropout=0.0):
            layers = [
                torch.nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()
            ]
            if dropout:
                layers.append(torch.nn.Dropout(dropout))
            return torch.nn.Sequential(*layers)

        self.down1 = down_block(3, 64, normalize=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)

        self.up1 = up_block(512, 512, dropout=0.5)
        self.up2 = up_block(512, 256, dropout=0.5)
        self.up3 = up_block(256, 128)
        self.up4 = up_block(128, 64)
        self.final = torch.nn.ConvTranspose2d(64, 3, 4, 2, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5)
        u2 = self.up2(u1 + d4)
        u3 = self.up3(u2 + d3)
        u4 = self.up4(u3 + d2)
        out = torch.tanh(self.final(u4 + d1))
        return out

# ------------------------------ Helper Functions ------------------------------
def extract_objects(image, results):
    objects = []
    img_h, img_w, _ = image.shape
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            obj = image[y1:y2, x1:x2]
            objects.append((obj, (x1, y1, x2, y2)))
    return objects

def remove_objects(image, results):
    img_h, img_w, _ = image.shape
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            mask[y1:y2, x1:x2] = 255
    return cv2.inpaint(image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

def blend_object(target_img, obj, coords):
    x1, y1, x2, y2 = coords
    img_h, img_w, _ = target_img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    obj_resized = cv2.resize(obj, (x2 - x1, y2 - y1))
    mask = 255 * np.ones(obj_resized.shape, obj_resized.dtype)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    if 0 <= center[0] < img_w and 0 <= center[1] < img_h:
        return cv2.seamlessClone(obj_resized, target_img, mask, center, cv2.NORMAL_CLONE)
    else:
        return target_img

# ------------------------------ Streamlit UI ------------------------------
st.set_page_config(page_title="CycleGAN + Pix2Pix + Stable Diffusion Inpainting", layout="wide")
st.title("🔁 CycleGAN + Pix2Pix + Stable Diffusion Inpainting")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Input Image", width=300)

    if st.button("🔮 Generate & Replace Objects"):
        with st.spinner("Running full pipeline..."):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = Generator().to(device)
            model.load_state_dict(torch.load("G_AB_epoch205.pth", map_location=device))
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output_tensor = model(input_tensor)
            output_tensor = output_tensor.squeeze().cpu()
            output_tensor = (output_tensor * 0.5) + 0.5
            generated_image = transforms.ToPILImage()(output_tensor)
            st.image(generated_image, caption="🌀 After CycleGAN (Generated Image)", width=300)
            generated_cv2 = cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2BGR)
            cv2.imwrite("generated2.jpg", generated_cv2)

            img1 = np.array(image)[:, :, ::-1].copy()
            img2 = np.array(generated_image)[:, :, ::-1].copy()

            yolo_model = YOLO("yolov8n.pt")
            results1 = yolo_model.predict(img1, conf=0.37)
            results2 = yolo_model.predict(img2, conf=0.37)
            results3 = yolo_model.predict(generated_cv2, conf=0.37)

            annotated1 = results1[0].plot()
            annotated2 = results2[0].plot()
            st.image(annotated1, caption="📍 Object Detection on Original Image", channels="BGR")
            st.image(annotated2, caption="📍 Object Detection on Generated Image", channels="BGR")

            extracted_objects = extract_objects(img1, results1)
            output_img = remove_objects(img2, results2)
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), caption="🧼 After Removing Detected Objects", channels="RGB")

            if extracted_objects:
                obj, coords = extracted_objects[0]
                final_img = blend_object(output_img, obj, coords)
                st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption="🔁 After Object Blend", channels="RGB")
                # cv2.imwrite("generated2.jpg", final_img)

                # === Pix2Pix Enhance ===
                pix2pix_model = UNetGenerator().to(device)
                pix2pix_model.load_state_dict(torch.load("generator_latest.pth", map_location=device))
                pix2pix_model.eval()

                final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                pil_final = Image.fromarray(final_img_rgb)
                input_pix2pix = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor()
                ])(pil_final).unsqueeze(0).to(device)

                with torch.no_grad():
                    pix2pix_output = pix2pix_model(input_pix2pix)

                output_image = pix2pix_output.squeeze().cpu()
                output_image = (output_image + 1) / 2
                pix2pix_final_image = transforms.ToPILImage()(output_image)
                pix2pix_final_image.save("pix2pix_final.jpg")

                # === Stable Diffusion Inpainting ===
                mask = np.zeros(img2.shape[:2], dtype=np.uint8)
                for result in results2:
                    for box in result.boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box.tolist())
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img2.shape[1], x2), min(img2.shape[0], y2)
                        cv2.rectangle(mask, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), 255, -1)
                cv2.imwrite("mask.png", mask)
                mask = np.zeros(generated_cv2.shape[:2], dtype=np.uint8)
                for result in results3:
                    for box in result.boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box.tolist())
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(generated_cv2.shape[1], x2), min(generated_cv2.shape[0], y2)
                        cv2.rectangle(mask, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), 255, -1)

                cv2.imwrite("mask_sd.png", mask)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                ).to("cuda" if torch.cuda.is_available() else "cpu")

                result_sd = pipe(
                    prompt="A modern clean bedroom with minimal decor and soft lighting",
                    image=Image.open("generated2.jpg").convert("RGB"),
                    mask_image=Image.open("mask_sd.png").convert("L")
                ).images[0]
                result_sd.save("final_output_sd_only.jpg")

                # === Display Both Outputs ===
                col1, col2 = st.columns(2)
                with col1:
                    st.image("pix2pix_final.jpg", caption="🧠 Pix2Pix Final Output", use_container_width=True)
                with col2:
                    st.image("final_output_sd_only.jpg", caption="🎨 Stable Diffusion Final Output", use_container_width=True)
            else:
                st.warning("No objects detected in input image.")
