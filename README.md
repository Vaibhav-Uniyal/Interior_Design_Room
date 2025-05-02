# 🏠 AI-Powered Interior Design Generator using GANs


## 📌 Project Overview

This project leverages *Generative Adversarial Networks (GANs)* and *Computer Vision* to generate realistic interior design layouts based on user preferences. It aims to bridge the gap between design imagination and visual execution, empowering individuals and interior designers to visualize room designs without manual effort or high costs.

---

## 🚀 Features

* ✨ *AI-Generated Room Designs* based on user input (style, theme, layout)
* 🎨 *Style Customization* (Modern, Minimalist, Vintage, etc.)
* 🧠 *GAN-based Image Generation* for realistic interiors
* 🔄 *Before/After Room Transformations*
* 📷 *Image-to-Design Pipeline* for real room enhancement
* 📊 *Impact & Feedback Tracking System* (Optional Extension)

---

## 📚 Technologies Used

* *Python*
* *PyTorch*
* *StyleGAN / AttnGAN*
* *OpenCV / PIL*
* *Streamlit* (for interface)
* *NumPy, Pandas* (for data handling)
* *Matplotlib / Seaborn* (for visualization)

---

## 🧠 Methodology

1. *Input Collection*: Style/theme preferences or base room image
2. *Preprocessing*: Resize, normalize, and prepare input
3. *GAN Training*: Train on dataset of annotated room layouts
4. *Generation*: Use generator model to create interior designs
5. *Evaluation*: Realism scored via FID, Inception Score & user feedback
6. *Deployment*: Simple UI using Streamlit

---

## 📂 Dataset

* *Source*: LSUN bedroom scene 20% sample
* *Size*: 303125 jpgs containing bedroom scenes
* *Annotations*: Furniture

---

## 📈 Results

* Successfully generated diverse interior designs
* High realism score (FID < threshold)
* Positive feedback on style accuracy and room relevance
* Deployment-ready web interface for demo purposes

---

## ✅ How to Run

1. Clone the repo

   bash
   git clone https://github.com/yourusername/interior-design-gan.git
   
2. Navigate to project folder

   bash
   cd interior-design-gan
   
3. Install requirements

   bash
   pip install -r requirements.txt
   
4. Run Streamlit app

   bash
   streamlit run app.py
   

---

## 🔒 Ethical Considerations

* Ensuring AI doesn't replicate biased design standards
* Respecting cultural diversity in training data
* Promoting sustainable and inclusive design suggestions

---

## 🔮 Future Scope

* AR integration for live room visualization
* Real-time customization and drag-drop editing
* Eco-friendly design suggestions using AI inference
* Collaboration tools for professionals and clients

---

## 🤝 Contributors

* Vaibhav Uniyal
* Sharry Dhiman 
* Vidisha Sharma
* Tanishq Jain

  
---

## 📬 Contact

For any questions or collaborations:
* 📧 [vaibhavuniyal10@gmail.com](mailto:vaibhavuniyal10@gmail.com)
🔗 \[www.linkedin.com/in/vaibhavuniyal10]

* 📧 [sharry@example.com](mailto:vidisha.sharma@example.com)
🔗 \[LinkedIn Profile]

* 📧 [vidisha@example.com](mailto:vidisha.sharma@example.com)
🔗 \[LinkedIn Profile]

* 📧 [tanishq@example.com](mailto:vidisha.sharma@example.com)
🔗 \[LinkedIn Profile]

---
