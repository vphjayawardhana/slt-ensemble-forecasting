# AI Usage Statement

**Tool.** Claude (Anthropic) was used as a coding and writing assistant through a chat interface during this project.

**Student's role.** I 224237K defined the assignment, chose the dataset and the research question, and made the methodological decisions: forecasting the next trading observation, a chronological 70/15/15 split with no shuffling, a leakage-safe design, comparing bagging, boosting and stacking against a persistence baseline, and using walk-forward validation for stacking. Claude gave suggestions and options, such as predicting the log return instead of the raw price. I reviewed each suggestion and approved it before it was implemented. I approved each phase before the next one began.

**What the AI did.** Claude wrote the Python code in `src/`, the Jupyter notebook and the Streamlit dashboard, wrote the verification scripts, and drafted the explanatory text in the notebook and the documentation. I reviewed this text.

**What I ran myself.** I ran preprocessing, model training and verification on my own MacBook Air M1 (Python 3.11). The models and results used by the notebook and dashboard come from that run. When I noticed that the hyperparameters selected on my Mac differed from the AI's reference run, I asked for the cause to be investigated.

**Safeguards against AI error.** No synthetic data was used; only the supplied daily CSV. Tables and figures are produced by code that reads the saved models and results. Scripts check for data leakage and reload the saved models to reproduce the reported metrics. Claude ran its own checks in a Linux sandbox; those results are reference only and were not used as the reported results.

**Limits.** The AI could not run anything on my Mac. Its explanations of *why* the results look as they do are interpretations, not proven causes. I am responsible for understanding and being able to explain all of the code and text.
