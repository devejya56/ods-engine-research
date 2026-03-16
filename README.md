# ODS Engine: Leading-Indicator Early Stopping for Deep Learning

The **Overfitting Detection Score (ODS)** is a domain-agnostic engine designed to identify the critical transition point from **feature learning** to **sample memorization**. Traditional early stopping relies on validation loss (a lagging indicator), whereas ODS utilizes structural signals within the network to provide a high-sensitivity "leading indicator" of overfitting.

## 🚀 Key Functionalities
- **Leading Indicator Stopping**: Triggers early stopping *before* the validation loss diverges.
- **Multi-Signal Tracking**: Integrates Loss Curvature, Gradient Norm Dynamics, and Prediction Confidence Divergence.
- **Adaptive Thresholding**: Auto-tunes the stopping sensitivity based on architecture-specific noise during the warmup phase.
- **Architecture Agnostic**: Validated on CNNs, ResNets, MLPs, and Vision Transformers (ViTs).

## 🛠️ How it's Built
The project is built as a modular Python package (`ods_engine`) designed for seamless integration with PyTorch:
- **`ODSCore`**: The mathematical heart of the system, implementing smoothed signal products and adaptive thresholding.
- **`Trackers`**: Automated hook-less metric extraction for loss, weights/gradients, and softmax distributions.
- **`ODSConnector`**: A high-level wrapper that integrates into standard training loops with minimal code changes.

## 🧪 Experimental Validation
We documented ODS's performance across 20+ research phases:
- **Medical Imaging (Pneumonia Detection)**: Demonstrated robustness on high-stakes clinical data.
- **Architecture Scaling**: Proved effective across **CNN**, **ResNet-18**, **MLP**, and **Vision Transformers**.
- **Efficiency Wins**: Achieved up to **33% faster training** cycles by stopping at the precise moment of peak generalization.
- **Accuracy Gains**: Prevented a **6% accuracy drop** on MNIST compared to conventional methods.

## 📈 Usecases
- **Medical AI**: Where overfitted models can lead to dangerous "shortcut learning" in diagnostics.
- **Small Dataset Learning**: Where the window between learning and overfitting is extremely narrow.
- **High-Capacity Models**: Helping Transformers and Deep ResNets stop before they waste compute on memorization.

## 📐 Mathematical Foundation
ODS calculates a composite score $S_t$:
$$S_t = \text{SMOOTH}(L_t \times G_t \times C_t)$$
Where:
- $L_t$: Second-order loss curvature.
- $G_t$: Scale-invariant normalized gradient norm.
- $C_t$: Divergence between training and validation confidence distributions.

---
*For a deep dive into the research and methodology, see [docs/ods_research_paper.md](docs/ods_research_paper.md).*
