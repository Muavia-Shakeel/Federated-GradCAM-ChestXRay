# Project Explanation: Federated Learning with Explainable AI for Chest X-Rays

This document explains our machine learning research project in simple terms. It is designed for team members without an AI background to understand what we are building, why it matters, and how it is different from existing research.

---

## 1. Core Concepts (Terminologies Explained)

To understand our project, you only need to understand three main concepts:

### A. Machine Learning for Chest X-Rays
We are training a computer program (an AI model called EfficientNet-B0) to look at chest X-ray images and predict if the patient has any of 14 different chest diseases (like Pneumonia, Hernia, Emphysema, etc.). 

### B. Federated Learning (FL) - The Privacy Shield
Normally, to train a smart AI, you need to collect thousands of X-rays from different hospitals into one central server. **Problem:** Patient data is highly confidential, and hospitals are not allowed to share it due to privacy laws (like HIPAA).
**Solution (Federated Learning):** Instead of moving the *data* to the AI, we send the *AI* to the data. 
1. We send a copy of the AI model to Hospital A, Hospital B, and Hospital C.
2. Each hospital trains the AI locally on its own private data.
3. The hospitals only send the "learnings" (mathematical weight updates, not patient images) back to a central server.
4. The central server combines (aggregates) these learnings to create one super-smart Global Model. 

### C. Explainable AI (XAI) and GradCAM - The "Why"
AI is often a "Black Box". If the AI says a patient has Pneumonia, a doctor will ask, *"Why? Show me where you see it."* If the AI can't explain itself, doctors won't trust it.
**Solution (GradCAM):** GradCAM is a technique that acts like a thermal camera for the AI's brain. It generates a "Heatmap" over the X-ray image. The red/hot areas show exactly which pixels the AI looked at to make its decision.

---

## 2. The Problem We Are Solving (Research Gap)

Researchers have done Federated Learning before. Researchers have also done GradCAM (heatmaps) before. **But combining them properly hasn't been done.**

**The Flaw in Existing Systems:**
In current research, when multiple hospitals train a Federated model, they only combine the *model weights* at the central server. If they want an explanation (heatmap), they just run the final model at the end. 
But hospitals are different! A rural clinic might have 200 X-rays of mostly healthy people. A major city hospital might have 10,000 X-rays of severe diseases. 
If we just look at the final model's explanation, we lose the unique diagnostic perspectives of the individual hospitals.

---

## 3. Our Unique Innovation (How We Are Different)

Our project introduces a novel technique called:
**"Dataset-Size-Weighted Federated GradCAM Aggregation"**

Here is what we do differently:
1. **Aggregating Explanations, Not Just Weights:** During training, we don't just ask the hospitals for their mathematical weights. We also ask them for their local GradCAM heatmaps.
2. **Fairness Based on Data Size (Weighted Aggregation):** When the central server combines these heatmaps to create a "Global Explanation", it gives more importance (weight) to the hospitals that have more data. If Hospital A has 80% of the data, their heatmap contributes 80% to the final global heatmap.
3. **Measuring Quality (Quantitative XAI):** Previous papers just look at heatmaps visually and say "it looks good." We are writing code to mathematically measure *how accurate* our aggregated heatmaps are using advanced metrics (Faithfulness, AOPC, Insertion/Deletion AUC).

**Summary of our difference:**
- **Others:** Federated Learning (Privacy) ✅ + No Explainability ❌
- **Others:** Centralized Explainability ✅ + No Privacy ❌
- **Others:** Federated Learning ✅ + Post-hoc Global Explanation (Not Fair) ❌
- **OURS:** Federated Learning (Privacy) ✅ + Fair, Dataset-Weighted Global Explanations ✅ + Mathematical Evaluation ✅

---

## 4. What Does the Code Actually Do?

Our Python code (`src/` folder) does the following automatically:
1. **Data Prep:** Takes 112,000 real Chest X-rays from the NIH dataset and mathematically splits them unevenly among 5 "virtual hospitals" to simulate real life (this is called Non-IID partitioning).
2. **Training:** Trains the AI for 20 rounds. In each round, the 5 hospitals train locally, then send their updates to the server.
3. **Aggregation:** The server combines the updates (FedAvg algorithm) AND combines the heatmaps using our novel Dataset-Size-Weighted method.
4. **Testing & Results:** At the end, the code tests the model on unseen data, calculates accuracy (AUC, F1-score), and measures the quality of our generated heatmaps.

---

## 5. Summary for Your Presentation

If you need to explain this in 30 seconds:
> *"We built an AI that diagnoses chest diseases without violating patient privacy by using Federated Learning. Our main innovation is that we created a fair way to combine visual explanations (heatmaps) from different hospitals, giving more voting power to hospitals with more data. We proved mathematically that our method generates trustworthy explanations that doctors can rely on."*
