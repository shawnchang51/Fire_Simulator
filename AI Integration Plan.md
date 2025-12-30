# Integration plan — Pairwise ranking AI as a design-evaluator for the evacuation simulator

We integrate an AI ranking model as a *search accelerator* that narrows candidate door configurations before the simulator performs expensive Monte-Carlo evaluation. The simulator remains the single ground-truth oracle; the AI learns relative preferences from simulator comparisons and is used only to pre-screen candidates. This keeps risk low, training efficient, and results explainable.

**Components**

* *Design Candidate Generator*: rule-based / random sampler that produces legally constrained layouts (doors on walls, fixed counts). Produces many complete candidates per floorplan.
* *Scoring Network (CNN + optional GNN fusion)*: a feature extractor that maps a layout (grid + adjacency) to a scalar latent score. The network is trained with pairwise comparison labels.
* *Pairwise Labeler (Simulator Oracle)*: component that runs k Monte-Carlo runs per candidate and produces pairwise “A > B?” supervision signals (using median/robust comparator).
* *Training Pipeline*: pair sampler, batcher, pairwise loss function, optimizer, checkpointing, and logging.
* *Inference / Search Controller*: uses the trained scoring network to rank large candidate pools and select top-k for final simulation.
* *Evaluation Dashboard*: visualizations (score vs. simulator metric scatter, Spearman, top-k recall, sim calls saved, case study visuals).

**Training dataflow (conceptual)**

1. For a set of floorplans, generate N candidates each using the Candidate Generator.
2. Sample pairs (A,B) from each candidate pool according to an exploration mix (hard pairs near predicted boundary + easy pairs far apart).
3. For each candidate in sampled pairs, run k (e.g., 3–5) simulator trials and compute a robust summary (median or trimmed mean).
4. Convert scores to pairwise labels: label = 1 if score_A > score_B + margin, else 0; discard ambiguous pairs if desired.
5. Train Scoring Network to minimize pairwise loss (logistic / hinge) so s(A) > s(B) when label = 1. Keep scoring head as a scalar latent value.
6. Monitor correlation metrics between model score and simulator metric, plus ranking metrics (Spearman, top-k recall).

**Inference dataflow (conceptual)**

1. For a new floorplan, generate a large candidate pool (N_large).
2. Run Scoring Network forward on all candidates to produce latent scores.
3. Take top-k candidates by score (k chosen by sim budget) and run full Monte-Carlo evaluation on those only.
4. Present top validated designs (and their simulator metrics) in the dashboard; optionally show the model’s heatmap or top candidate to demo “AI suggested” design.

**Practical choices & robustness**

* Use pairwise ranking (not direct regression) because simulator outputs are noisy; pairwise labels are robust to variance.
* Use median or trimmed-mean from k runs to reduce label noise; set k to 3–5 during training for speed.
* Use a mix of random and model-guided candidate generation during training to avoid collapse.
* Optionally include a Proposal Head (CNN heatmap) to bias sampling: train it jointly with ranking via auxiliary losses (connectivity/distance) but keep the main supervision pairwise.
* Prevent hard non-differentiable ops in the training graph (we do not backprop through the simulator).

**Metrics to report**

* Spearman rank correlation between model score and simulator metric across held-out plans.
* Top-k recall / precision: fraction of true top-X designs found within top-k model selections.
* Simulation calls saved: ratio of sim calls required by AI-guided search vs. random search to reach comparable top solutions.
* Case studies with visual comparisons (random vs. AI-selected layouts + corresponding sim outcomes).

**Deliverables (demo-ready, within short timeline)**

* Trained Scoring Network checkpoints and inference script.
* Candidate generator and inference controller (selects top-k, runs simulator).
* Dashboard with scatter plot (model score vs. simulator), ranking metrics, and 3 case studies (visual + numbers).
* One or two “AI suggested” layouts per demo floorplan (show both model score and simulator-validated metric).
* Short methods blurb explaining: pairwise training, simulator-as-oracle, and why this approach is robust and extensible.

**Integration notes**

* Keep the AI module as a plug-in service (HTTP or local function) that accepts floorplan + candidate and returns score; simulator remains unchanged.
* Log all candidate → score → simulator outcomes to allow later fine-tuning or switching to generate-and-score loop if more budget becomes available.
* Emphasize in presentation that AI *narrows search* and *enables efficient use of simulator budget*; the simulator remains the final decision maker.

---

This plan yields a compact, defensible AI component: it demonstrates that your simulator can support learning-based design evaluation, it produces measurable search-speed gains, and it preserves interpretability and reproducibility for a short project deadline.
