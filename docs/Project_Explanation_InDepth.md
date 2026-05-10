# Comprehensive Project Guide: Federated Learning with Explainable AI for Chest X-Rays

This document provides an in-depth, easy-to-understand breakdown of our research project. It is written specifically for team members without a formal AI or computer science background, explaining every terminology, the problem we are solving, the existing research landscape, and our unique innovation.

---

## 1. Introduction: The Problem with Medical AI
Artificial Intelligence (AI) can diagnose diseases from medical images (like X-Rays) with incredible accuracy. However, two major roadblocks prevent hospitals from using AI in real life:
1. **Privacy Laws (HIPAA):** Hospitals cannot share patient X-rays with central tech companies to train AI. It is illegal and unethical.
2. **Lack of Trust (The Black Box Problem):** If an AI tells a doctor, "This patient has a 90% chance of Pneumonia," the doctor will ask, "Why? Show me the evidence." Standard AI cannot explain itself; it acts as a "Black Box". Doctors cannot trust a system that cannot explain its reasoning.

Our project solves **both** problems simultaneously.

---

## 2. In-Depth Terminologies

To understand the project, you need to know these five core concepts:

### A. EfficientNet-B0 (The "Brain")
This is the specific type of AI model (Convolutional Neural Network) we are using. Think of it as a virtual brain specifically designed to look at images. We chose "B0" because it is lightweight and efficient, meaning hospitals don't need supercomputers to run it.

### B. Multi-Label Classification
In real life, a patient can have more than one disease at a time. Multi-label classification means our AI doesn't just say "Healthy" or "Sick". It looks at 14 different diseases (e.g., Asthma, Emphysema, Hernia) and gives a probability for *each one simultaneously*.

### C. Federated Learning (FL) - The Privacy Shield
Federated Learning is how we train the AI without stealing patient data. 
- Instead of moving private patient data to a central server, we send the "untrained AI" to the hospitals (Clients).
- The AI learns locally at each hospital.
- After learning, the AI only sends back its "mathematical knowledge" (called Weights) to the central server, not the patient X-rays.
- The server combines the knowledge from all hospitals to create a "Global Master AI". 

### D. Non-IID Data (Real-World Simulation)
"Non-IID" stands for Non-Independent and Identically Distributed. In simple terms: **Hospitals are not identical.** 
- Hospital A might be a rural clinic with mostly healthy patients.
- Hospital B might be a specialized lung cancer center with severe cases.
- Hospital C might have 10,000 patients, while Hospital A only has 500.
We programmed our code to simulate this real-world imbalance. This makes our project much more realistic than standard research.

### E. Explainable AI (XAI) & GradCAM
Explainable AI is the field of making AI understandable to humans. **GradCAM** is our chosen technique. It creates a "Heatmap" (like a thermal camera) over the X-ray. 
- **Red/Hot areas:** The exact pixels the AI stared at to diagnose the disease.
- **Blue/Cold areas:** Background noise the AI ignored.
GradCAM gives doctors the visual proof they need to trust the AI.

---

## 3. The Literature Review: Where Past Research Failed

In our Assignment 3, we reviewed 8 to 10 major research papers in this field. Here is why our work is different and better:

### Group 1: The Privacy-Only Papers (e.g., Kaissis, Adnan, Sheller)
These researchers successfully used Federated Learning to protect patient privacy. 
* **Their Flaw:** They completely ignored Explainability. Their final models are perfect black boxes. A doctor using their system gets a diagnosis but no visual heatmap. It is legally compliant but clinically useless.

### Group 2: The Explainability-Only Papers (e.g., Selvaraju, Tjoa)
These researchers invented and tested GradCAM and other heatmap technologies.
* **Their Flaw:** They assumed the AI has a "Centralized" database with all patient data in one place. They did not design their explanations for a Federated (distributed) privacy-first world.

### Group 3: The Flawed Combinations (e.g., 2024 Blockchain & FL Paper)
A few recent papers tried to combine Federated Learning and GradCAM. 
* **Their Flaw:** They used "Post-Hoc" explanation. This means they trained the global model, and at the very end, they just generated a heatmap. They ignored the fact that Hospital A and Hospital B might have looked at the disease differently! By only looking at the final global model, they erased the unique diagnostic perspectives of the individual hospitals.

---

## 4. Our Core Innovation: What We Built

We created a brand new technique called:
**"Dataset-Size-Weighted Federated GradCAM Aggregation"**

Instead of just combining the "brain weights" of the AI, our central server also securely collects the "heatmaps" from every hospital during the training process. 

**How it works (The Innovation):**
1. Hospital A (10,000 patients) generates a heatmap.
2. Hospital B (500 patients) generates a heatmap.
3. When the Central Server combines these two heatmaps to create a "Global Explanation", it uses mathematics to give **20 times more voting power** to Hospital A's heatmap because they have 20 times more experience (data). 

This is incredibly important because it creates a **Fair, Democratized Global Explanation**. Without this sizing weight, a tiny rural clinic with 5 weird cases could completely ruin the global heatmap.

---

## 5. How We Prove It Works (Quantitative Evaluation)

In the past, researchers would just look at a heatmap and say, "Looks good to me." In modern AI, that is not enough. Our code mathematically proves our heatmaps are accurate using three advanced metrics:

1. **Faithfulness Score:** We tell the computer to black out the "red/hot" pixels from the X-ray and feed it back to the AI. If the AI's confidence drops massively, it proves that those red pixels were indeed the most important.
2. **Insertion AUC:** We start with a blank image and slowly reveal the red pixels. If the AI quickly makes a correct diagnosis, our heatmap is highly accurate.
3. **Deletion AUC:** We slowly delete the red pixels. If the AI's prediction quickly collapses, it proves our heatmap successfully identified the critical disease markers.

No other paper has applied these strict mathematical XAI metrics to a Federated Learning chest X-ray system.

---

## 6. What the Code Actually Does

If you look at our project folder, here is what our Python code is doing step-by-step:
1. **`dataset.py` & `partition.py`:** Loads 112,000 X-rays and splits them unevenly (Non-IID) into 5 virtual hospitals.
2. **`main.py`:** Runs the main simulation. It loops 20 times (called Rounds).
3. **`train_client.py`:** Simulates the hospitals training their local AI brains on their private data.
4. **`gradcam_aggregation.py`:** *This is our secret sauce.* It takes the local heatmaps and averages them based on hospital dataset size.
5. **`metrics.py`:** Runs the Faithfulness, Insertion, and Deletion math at the end to generate our final result scores.

---

## 7. Conclusion

By sharing this document, anyone can understand that our project is not just running an AI. We are solving a critical gap in medical AI literature: **Providing trustworthy, mathematically verified visual explanations in a strictly private, multi-hospital environment.**
