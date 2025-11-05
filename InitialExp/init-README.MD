Initial Experiments  27th - 13th Nov
Summary of the Presentation
The team presented their plan to use prompt engineering, rather than supervised fine-tuning (SFT), to optimize model responses for customer queries. They justified this approach by highlighting prompt engineering’s scalability, speed, and cost-efficiency compared to SFT, which would require significant resources and labeled data. The team identified key hyperparameters for prompt engineering (tone, accuracy, creativity, etc.) and proposed a hybrid evaluation method combining human and automated assessments. They also described their process for sourcing and cleaning a dataset of customer queries and responses, and outlined plans to benchmark various models (GPT-4.0, 4.0 Mini, 3.5 Turbo) using a custom Python script and spreadsheet for evaluation
My Feedback
You agreed with the prompt engineering focus, noting it should take no more than 1-2 weeks to reach an ideal prompt.
You encouraged expanding the project’s scope for greater learning, suggesting possible future work with Retrieval-Augmented Generation (RAG).
You emphasized the need for more research on model response times and trade-offs, and suggested including faster models (like Nano).
You requested more nuanced evaluation metrics, especially regarding accuracy (brevity, relevance, clarity), and a clear business case for model selection.
You asked for a formula for hybrid evaluation in the final plan and stressed the importance of relating technical work back to business value.
You want to see initial experiments, a partial dataset, and a revised plan reflecting learnings at the next checkpoint.
Detailed Approach: Deliverables and Next Steps
Initial Experimentation & Data Preparation
Begin by running initial experiments with prompt engineering using the cleaned dataset of customer queries and responses.
Ensure the dataset contains high-quality question-answer pairs; consider both short and long responses, and refine selection criteria using sentiment analysis or manual review.
Document the data cleaning process and rationale for inclusion/exclusion of data.
Model Benchmarking & Evaluation
Implement a benchmarking pipeline using a Python script to send prompts to selected models (e.g., GPT-4.0, 4.0 Mini, Nano, 3.5 Turbo), record responses, and measure latency.
Evaluate model outputs using both human scoring (accuracy, tone, relevance, clarity, brevity) and automated metrics (sentiment, response length, latency).
Develop and document a hybrid evaluation formula that combines human and automated scores, with clear weighting rationale.
Business Case & Metric Definition
Clearly articulate the business case for model selection, focusing on cost, speed, and response quality.
Define what constitutes a “good” response in business terms, breaking down accuracy into subcomponents like brevity and clarity.
Iterative Improvement & Reporting
After initial experiments, analyze results and iterate on prompt design, evaluation metrics, and model selection.
Prepare a short presentation or pitch deck summarizing:
Key findings from initial experiments
Adjustments made to the approach
Updated plan for the remaining weeks
Include evidence of platform usage (e.g., API call logs, cost tracking).
Next Checkpoint Deliverables
A partial but functional dataset and initial experiment results.
A draft of the hybrid evaluation formula.
A revised plan for the next three weeks, reflecting learnings and outlining steps to final delivery.
A clear summary slide or document for each of the above.

