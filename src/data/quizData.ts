export const fullQuizData = [
  {
    question: `What is one of the main goals of Explainable and Trustworthy AI?`,
    options: [
      "Maximize the computational performance of the model",
      "Make models interpretable and promote ethical use of AI",
      "Reduce the amount of data required for training",
      "Ensure complete data anonymization"
    ],
    correct: "b",
    explanation: `The goal of XAI is to ensure transparency, ethics, interpretability, and trust across the full lifecycle of an AI system.`,
    topic: "Explainable and Trustworthy AI"
  },
  {
    question: `What characterizes explainability in the pre-modeling phase?`,
    options: [
      "Replacing the dataset with synthetic data",
      "Using complex models to produce early predictions",
      "Analyzing and transforming data in an interpretable way before modeling",
      "Optimizing the black-box model for production"
    ],
    correct: "c",
    explanation: `Pre-modeling explainability refers to the activities prior to modeling: exploration, documentation, and interpretable feature engineering.`,
    topic: "Explainable and Trustworthy AI"
  },
  {
    question: `Which of the following practices is typically associated with EDA (Exploratory Data Analysis)?`,
    options: [
      "Automatically modifying target values",
      "Training the final model",
      "Analyzing descriptive statistics and detecting outliers",
      "Deploying to production environments"
    ],
    correct: "c",
    explanation: `EDA includes statistical analysis, visualization, and anomaly detection in the data.`,
    topic: "Explainable and Trustworthy AI"
  },
  {
    question: `What is meant by "interpretable feature engineering" and why is it important in the pre-modeling phase?`,
    expected_answer: `Interpretable feature engineering means transforming or selecting features so they remain understandable to humans. It is important because it improves transparency and model robustness from the early stages.`,
    topic: "Explainable and Trustworthy AI"
  },
  {
    question: `List two main benefits of introducing explainability into an AI system.`,
    expected_answer: `Two core benefits are: (1) Increased user trust, as decisions become understandable; (2) Ability to detect bias or errors in the data or model.`,
    topic: "Explainable and Trustworthy AI"
  },
  {
    question: `Discuss the importance of the pre-modeling phase for ensuring the explainability of an AI system. What tools or techniques can improve it?`,
    expected_answer: `The pre-modeling phase is crucial because it lays the foundation for future model transparency. It includes EDA (statistical and visual data exploration), documentation (datasheets for datasets, nutrition labels), and interpretable feature engineering (e.g., discretization, domain knowledge). Tools like YData-Profiling or FACETS help reveal structure, anomalies, and distributions. A well-designed pre-modeling stage improves both predictive power and user trust.`,
    topic: "Explainable and Trustworthy AI"
  },
  {
    question: `Which of the following phases is NOT typically part of the XAI taxonomy?`,
    options: [
      "Pre-modeling",
      "Interpretable modeling",
      "Post-deployment explainability",
      "Post-modeling"
    ],
    correct: "c",
    explanation: `The standard taxonomy includes Pre-modeling, Interpretable Modeling, and Post-modeling. Post-deployment explainability is not a formal category, although it's relevant in practice.`,
    topic: "XAI Taxonomy"
  },
  {
    question: `What does a global explanation in XAI refer to?`,
    options: [
      "An explanation focused on a single prediction",
      "A rule-based local explanation",
      "A description of the overall model behavior",
      "A technique specific to neural networks"
    ],
    correct: "c",
    explanation: `Global explanations provide insight into the model’s behavior across the entire data domain.`,
    topic: "XAI Taxonomy"
  },
  {
    question: `Which of the following techniques belongs to the "explanation generation mechanism"?`,
    options: [
      "Gradients",
      "Batch Normalization",
      "Dropout",
      "Early Stopping"
    ],
    correct: "a",
    explanation: `Explanation mechanisms may include: removal, local surrogates, gradients. The others are training techniques.`,
    topic: "XAI Taxonomy"
  },
  {
    question: `Explain the difference between model-agnostic and model-specific methods in the XAI taxonomy.`,
    expected_answer: `A model-agnostic method can be applied to any model type as a black box (e.g., LIME, SHAP). A model-specific method is tailored to a specific architecture (e.g., gradients for neural networks).`,
    topic: "XAI Taxonomy"
  },
  {
    question: `What is meant by the "scope of explanation" in the XAI taxonomy?`,
    expected_answer: `It is one of the taxonomy dimensions and refers to the generality level of the explanation: global (entire model), subgroup (clusters), or local (single prediction).`,
    topic: "XAI Taxonomy"
  },
  {
    question: `Analyze the “form of explanation” dimension in XAI taxonomy. What are possible forms and why is it important to choose them wisely?`,
    expected_answer: `Forms include: (1) Feature importance, (2) Local symbolic rules, (3) Visualizations, (4) Counterfactual examples. Each form suits different users and use cases: counterfactuals help with actionability, visualizations are better for vision tasks. Choosing the right form enhances comprehension and impact.`,
    topic: "XAI Taxonomy"
  },
  {
    question: `What is the main purpose of Exploratory Data Analysis (EDA) in pre-modeling explainability?`,
    options: [
      "To automatically optimize model hyperparameters",
      "To clean the dataset using deep learning techniques",
      "To extract statistical and visual insights about the dataset before modeling",
      "To train an interpretable model"
    ],
    correct: "c",
    explanation: `EDA uses statistical and visualization techniques to understand the structure, anomalies, and patterns in data before training any model.`,
    topic: "Pre-Modeling and In-Modeling"
  },
  {
    question: `Which of the following is a characteristic of inherently interpretable models?`,
    options: [
      "They rely exclusively on neural network embeddings",
      "They use a hidden representation space with millions of parameters",
      "They produce predictions and explanations using transparent rules or formulas",
      "They do not allow for local explanations"
    ],
    correct: "c",
    explanation: `Inherently interpretable models like decision trees and linear models use understandable structures and formulas to generate both predictions and explanations.`,
    topic: "Pre-Modeling and In-Modeling"
  },
  {
    question: `What is the role of concept bottleneck models (CBM) in in-modeling explainability?`,
    options: [
      "They compress the model to improve performance",
      "They allow predictions to be made without labeled data",
      "They enforce intermediate human-interpretable concepts before the final prediction",
      "They operate only with unlabeled datasets"
    ],
    correct: "c",
    explanation: `CBMs add an interpretable layer of concepts between input and output, making predictions understandable and editable via concept interventions.`,
    topic: "Pre-Modeling and In-Modeling"
  },
  {
    question: `What is the purpose of documenting datasets during the pre-modeling phase?`,
    expected_answer: `Dataset documentation aims to increase transparency, reproducibility, and ethical awareness. It helps communicate data origin, structure, limitations, and potential biases.`,
    topic: "Pre-Modeling and In-Modeling"
  },
  {
    question: `Why is interpretable feature engineering important in pre-modeling explainability?`,
    expected_answer: `Because it ensures that the transformed features maintain human-meaningful semantics, making models more transparent and understandable.`,
    topic: "Pre-Modeling and In-Modeling"
  },
  {
    question: `Compare decision trees and linear models in terms of interpretability. How do these models support transparency during the modeling process?`,
    expected_answer: `Decision trees allow global and local interpretations via structure and paths. Linear models provide explicit equations where each coefficient shows the weight of a feature. Both are easy to explain to non-experts and support transparent decision making.`,
    topic: "Pre-Modeling and In-Modeling"
  },
  {
    question: `What is the primary goal of post-hoc global explainability techniques?`,
    options: [
      "To explain a single prediction for a specific input",
      "To describe the overall behavior of a trained model",
      "To replace the black-box model with an interpretable one",
      "To optimize training using local rules"
    ],
    correct: "b",
    explanation: `Post-hoc global explainability techniques aim to capture general patterns in a model's behavior to support model auditing, compliance, and trust.`,
    topic: "Post-Hoc Global Explainability"
  },
  {
    question: `What is the main idea behind global surrogate models?`,
    options: [
      "They inject noise into the training data to simulate variability",
      "They replace training features with synthetic ones",
      "They approximate the black-box model with an interpretable one trained on the same inputs and predictions",
      "They generate new features for the black-box model"
    ],
    correct: "c",
    explanation: `A surrogate model is trained to mimic the predictions of a black-box model using the same input data, making its logic easier to understand.`,
    topic: "Post-Hoc Global Explainability"
  },
  {
    question: `What does the Permutation Feature Importance (PFI) technique measure?`,
    options: [
      "The absolute correlation between each feature and the target",
      "The average marginal effect of each feature on the model output",
      "The change in model performance when a feature’s values are randomly shuffled",
      "The number of times a feature appears in decision rules"
    ],
    correct: "c",
    explanation: `PFI evaluates the importance of a feature by measuring how much the model's performance drops when that feature is randomly permuted.`,
    topic: "Post-Hoc Global Explainability"
  },
  {
    question: `What are Partial Dependence Plots (PDPs) used for?`,
    expected_answer: `PDPs visualize the average effect of one or more features on the model prediction by marginalizing over the other features.`,
    topic: "Post-Hoc Global Explainability"
  },
  {
    question: `Name one advantage and one limitation of global surrogate models.`,
    expected_answer: `Advantage: they are model-agnostic and interpretable. Limitation: they may oversimplify the true complexity of the black-box model, losing important patterns.`,
    topic: "Post-Hoc Global Explainability"
  },
  {
    question: `Compare and contrast Permutation Feature Importance and Partial Dependence Plots. How do these techniques help in understanding a black-box model globally?`,
    expected_answer: `PFI quantifies how much each feature impacts model performance by perturbing it, which helps in identifying the most influential features. PDPs show the average marginal effect of a feature on predictions, making it easier to interpret how a feature affects outputs. While PFI gives a ranking, PDPs offer a more detailed functional insight.`,
    topic: "Post-Hoc Global Explainability"
  },
  {
    question: `What is the core idea behind LIME (Local Interpretable Model-Agnostic Explanations)?`,
    options: [
      "Replacing the black-box model with a linear model",
      "Training a neural network to approximate global behavior",
      "Learning a simple model that approximates the black-box model locally around an instance",
      "Training the black-box model on interpretable data"
    ],
    correct: "c",
    explanation: `LIME builds an interpretable model around a specific instance to approximate how the black-box model behaves nearby.`,
    topic: "Local Surrogate Models"
  },
  {
    question: `Which of the following is a limitation of LIME?`,
    options: [
      "It cannot be applied to images or text",
      "It cannot be used with any black-box model",
      "It produces unstable explanations due to randomness in perturbation sampling",
      "It always requires labeled data for training"
    ],
    correct: "c",
    explanation: `LIME explanations may vary between runs because of random sampling of perturbed instances.`,
    topic: "Local Surrogate Models"
  },
  {
    question: `What distinguishes LORE from LIME?`,
    options: [
      "LORE uses global interpretable models",
      "LORE generates symbolic rules using local decision trees and genetic algorithms",
      "LORE modifies the architecture of the black-box model",
      "LORE only works with tabular data and ignores instance locality"
    ],
    correct: "b",
    explanation: `LORE generates local explanations using symbolic rules extracted from decision trees trained on synthetic data.`,
    topic: "Local Surrogate Models"
  },
  {
    question: `Why is it important that the perturbed instances in LIME are weighted by proximity to the original instance?`,
    expected_answer: `Because it ensures the surrogate model focuses on accurately approximating the black-box model in the local neighborhood of the instance to explain.`,
    topic: "Local Surrogate Models"
  },
  {
    question: `What are two key advantages of using LIME for local explainability?`,
    expected_answer: `It is model-agnostic (can be used with any black-box model) and supports various data types including text, images, and tabular data.`,
    topic: "Local Surrogate Models"
  },
  {
    question: `Explain the differences between LIME and LORE in how they generate explanations. What are the strengths and limitations of each approach?`,
    expected_answer: `LIME approximates the model locally using simple models like linear regression, based on perturbed samples. It is flexible and model-agnostic but may suffer from instability. LORE builds a local decision tree using synthetic data from genetic algorithms and provides symbolic rules and counterfactuals. It offers more structured explanations but is computationally more expensive.`,
    topic: "Local Surrogate Models"
  },
  {
    question: `What is the main idea behind local removal-based explanations?`,
    options: [
      "Replacing irrelevant features with synthetic data",
      "Training a global model to mimic the black-box",
      "Evaluating the model’s output variation when individual features are removed",
      "Using gradients to detect important input regions"
    ],
    correct: "c",
    explanation: `The key idea is that if removing a feature causes a significant change in prediction, that feature is important for the decision.`,
    topic: "Local Removal-Based Explainability"
  },
  {
    question: `What does a high value of ∆i = f(x) − f(x\\i) indicate?`,
    options: [
      "That feature xi is irrelevant for the prediction",
      "That removing feature xi decreases model accuracy",
      "That feature xi is highly important for the prediction of x",
      "That xi has been permuted successfully"
    ],
    correct: "c",
    explanation: `A high ∆i means the model output changes a lot without xi, indicating its importance for the current prediction.`,
    topic: "Local Removal-Based Explainability"
  },
  {
    question: `Which of the following is a valid masking/removal strategy used in local explanations?`,
    options: [
      "Replacing features with gradients",
      "Duplicating feature values",
      "Setting feature to zero, mean, or masking token depending on the data type",
      "Adding adversarial noise to all features"
    ],
    correct: "c",
    explanation: `Valid masking strategies include zeroing, replacing with mean/modality, or using [MASK] token in NLP tasks.`,
    topic: "Local Removal-Based Explainability"
  },
  {
    question: `What is the purpose of the Area Over the Perturbation Curve (AOPC)?`,
    expected_answer: `AOPC summarizes the overall impact of incrementally removing the most important features, offering a cumulative view of sensitivity.`,
    topic: "Local Removal-Based Explainability"
  },
  {
    question: `How are local removal-based explanations applied in image classification models?`,
    expected_answer: `By masking or blurring patches of the image to observe how the model's output changes, creating pixel-wise importance maps.`,
    topic: "Local Removal-Based Explainability"
  },
  {
    question: `Discuss the advantages and limitations of removal-based local explanations. When might they be especially useful or problematic?`,
    expected_answer: `Advantages: intuitive, model-agnostic, and do not require access to internal model layers. Limitations: may produce out-of-distribution inputs, can be computationally expensive, and sensitive to masking strategies. They're useful for high-stakes domains (e.g., healthcare) where clear causal importance is essential.`,
    topic: "Local Removal-Based Explainability"
  },
  {
    question: `What does a saliency map represent in gradient-based explanations?`,
    options: [
      "A histogram of prediction probabilities",
      "A visual highlight of input regions that most affect the model's output",
      "A summary of model training loss",
      "A list of important features for global model behavior"
    ],
    correct: "b",
    explanation: `Saliency maps visualize which parts of the input most influence the prediction, typically by computing the gradient of the output with respect to each input feature.`,
    topic: "Gradient-Based Local Explainability"
  },
  {
    question: `What does the Vanilla Gradient method compute in the context of XAI?`,
    options: [
      "The gradient of the loss with respect to model weights",
      "The second derivative of the output with respect to the input",
      "The gradient of the model output with respect to the input",
      "The change in activation maps between layers"
    ],
    correct: "c",
    explanation: `Vanilla Gradient computes the first-order derivative of the output for a given class with respect to each input feature.`,
    topic: "Gradient-Based Local Explainability"
  },
  {
    question: `Which of the following is a major limitation of Vanilla Gradient explanations?`,
    options: [
      "They require training a new model from scratch",
      "They cannot be computed for differentiable models",
      "They are unstable and sensitive to noise and saturated gradients",
      "They only apply to tabular data"
    ],
    correct: "c",
    explanation: `Vanilla gradients can be noisy and unreliable, especially in deep models due to issues like gradient saturation.`,
    topic: "Gradient-Based Local Explainability"
  },
  {
    question: `What is the purpose of computing the gradient of the output with respect to the input in explainability methods?`,
    expected_answer: `To measure how much small changes in each input feature would influence the model's prediction.`,
    topic: "Gradient-Based Local Explainability"
  },
  {
    question: `What are two advantages of using gradient-based explanation methods like saliency maps?`,
    expected_answer: `They are computationally efficient (only one backward pass needed) and applicable to any differentiable model.`,
    topic: "Gradient-Based Local Explainability"
  },
  {
    question: `Discuss the role of saliency maps in interpreting deep learning models. How are they computed and what are their strengths and weaknesses?`,
    expected_answer: `Saliency maps are computed by taking the gradient of the model's output with respect to the input features and visualizing their absolute values. They highlight which parts of the input most influenced the prediction. They are fast and simple, but sensitive to noise and can be hard to interpret. Their reliability is debated, and they are often used as a base for more robust methods like Integrated Gradients.`,
    topic: "Gradient-Based Local Explainability"
  },
  {
    question: `What is a Concept Bottleneck Model (CBM)?`,
    options: [
      "A model that uses only raw features for prediction",
      "A model that explains decisions using feature importance plots",
      "A model that predicts intermediate human-interpretable concepts before producing a final label",
      "A model that replaces concept labels with random embeddings"
    ],
    correct: "c",
    explanation: `CBMs structure the prediction pipeline by first predicting interpretable concepts and then using them to predict the output.`,
    topic: "Concept-Based Explainability"
  },
  {
    question: `What is a key innovation of Concept Embedding Models (CEMs)?`,
    options: [
      "They rely on unsupervised training only",
      "They learn two semantic embeddings per concept (active/inactive) for greater flexibility and interpretability",
      "They remove the concept layer entirely",
      "They require no annotations for training"
    ],
    correct: "b",
    explanation: `CEMs model each concept using two learnable vector embeddings and allow interventions via switching between them.`,
    topic: "Concept-Based Explainability"
  },
  {
    question: `Why are concept interventions important in CBMs and CEMs?`,
    options: [
      "They optimize hyperparameters during training",
      "They allow modifying predictions by changing concept values at test time",
      "They reduce model size",
      "They help generate synthetic data"
    ],
    correct: "b",
    explanation: `Concept interventions allow humans to directly manipulate model decisions through semantically meaningful units.`,
    topic: "Concept-Based Explainability"
  },
  {
    question: `How do CBMs help increase trust in AI systems?`,
    expected_answer: `By providing explanations based on human-interpretable concepts and enabling expert corrections (interventions) at test time.`,
    topic: "Concept-Based Explainability"
  },
  {
    question: `What are two limitations of traditional CBMs that CEMs try to overcome?`,
    expected_answer: `CBMs struggle when concept annotations are incomplete and may suffer from performance drops. CEMs improve generalization and enable richer embeddings without losing interpretability.`,
    topic: "Concept-Based Explainability"
  },
  {
    question: `Compare CBMs and CEMs in terms of interpretability, performance, and suitability for real-world applications. Why might a practitioner prefer CEMs?`,
    expected_answer: `CBMs are more interpretable due to their scalar concept predictions but may sacrifice accuracy when concepts are missing or noisy. CEMs learn richer vector representations for each concept and support test-time interventions with improved robustness. They outperform CBMs in incomplete settings (e.g., CelebA) and offer better trade-offs in real-world applications.`,
    topic: "Concept-Based Explainability"
  },
  {
    question: `What is a typical use of attention visualization in NLP explainability?`,
    options: [
      "To speed up token generation",
      "To identify which input tokens are most influential for a prediction",
      "To reduce model training time",
      "To hide the internal states of transformers"
    ],
    correct: "b",
    explanation: `Attention visualizations help interpret how each token influences others in generating a model's output, especially in transformer-based architectures.`,
    topic: "NLP Explainability"
  },
  {
    question: `Which of the following best describes the goal of interpretability in NLP?`,
    options: [
      "Optimizing the language model loss function",
      "Generating longer output sequences",
      "Making the inner workings and decisions of language models understandable to humans",
      "Using less training data"
    ],
    correct: "c",
    explanation: `Interpretability in NLP seeks to clarify how inputs affect outputs, often using attention maps, gradients, or concept-based techniques.`,
    topic: "NLP Explainability"
  },
  {
    question: `What is a limitation of using raw attention weights for explanation in transformer models?`,
    options: [
      "They are not available during inference",
      "They are expensive to compute",
      "They may not align with true model reasoning or token importance",
      "They can only be used for image inputs"
    ],
    correct: "c",
    explanation: `Attention weights don't always correlate with token importance or true model decision flow, leading to misleading explanations.`,
    topic: "NLP Explainability"
  },
  {
    question: `What does a saliency map in NLP typically show?`,
    expected_answer: `It shows which tokens in the input text most influenced the model’s prediction, based on gradient magnitude.`,
    topic: "NLP Explainability"
  },
  {
    question: `Why is interpretability especially important in NLP tasks like toxic comment classification?`,
    expected_answer: `Because it helps ensure that the model is not basing decisions on spurious correlations (e.g., identity terms), and supports transparency, fairness, and accountability.`,
    topic: "NLP Explainability"
  },
  {
    question: `Discuss the role of attention mechanisms in NLP models and their limitations as explanatory tools. How can alternative methods enhance interpretability?`,
    expected_answer: `Attention mechanisms enable models to focus on relevant parts of the input. However, attention weights can be diffuse or deceptive. Alternatives like gradient-based saliency or concept attribution (e.g., TCAV) provide complementary insights and improve reliability of explanations.`,
    topic: "NLP Explainability"
  },
  {
    question: `What does the concept of "faithfulness" in XAI evaluation refer to?`,
    options: [
      "The ability of the explanation to convince the user",
      "The clarity of the visual explanation layout",
      "How well the explanation reflects the actual internal behavior of the model",
      "The length of the explanation generated by the model"
    ],
    correct: "c",
    explanation: `Faithfulness measures the alignment between the explanation and the true reasoning process of the model. It includes completeness and correctness of highlighted features.`,
    topic: "XAI Evaluation"
  },
  {
    question: `What is the AOPC metric used for in explanation evaluation?`,
    options: [
      "To visualize which layer of the model is most interpretable",
      "To determine the time to compute an explanation",
      "To assess explanation consistency across models",
      "To evaluate how prediction confidence decreases as important features are removed"
    ],
    correct: "d",
    explanation: `AOPC (Area Over the Perturbation Curve) measures the impact of removing important features, assessing the relevance of the explanation to model predictions.`,
    topic: "XAI Evaluation"
  },
  {
    question: `Why might an explanation be plausible but not faithful?`,
    options: [
      "Because it is generated from gradients only",
      "Because it reflects the model’s internal reasoning exactly",
      "Because it makes sense to a human, even if it's not based on true model behavior",
      "Because it is too detailed to be misleading"
    ],
    correct: "c",
    explanation: `A plausible explanation may match human intuition but not reflect the true decision process of the model, potentially leading to overtrust or misinterpretation.`,
    topic: "XAI Evaluation"
  },
  {
    question: `What is meant by explanation "compactness"?`,
    expected_answer: `The explanation should be concise, focusing on a small set of meaningful features or rules to improve user comprehension.`,
    topic: "XAI Evaluation"
  },
  {
    question: `Name one user-centered metric and explain its relevance in XAI evaluation.`,
    expected_answer: `One example is "controllability", which measures how much a user can interact with or modify the explanation. It's relevant to assess how usable and actionable explanations are.`,
    topic: "XAI Evaluation"
  },
  {
    question: `Compare the concepts of faithfulness and plausibility in XAI evaluation. Why is it important to distinguish them, and how can this affect trust in AI systems?`,
    expected_answer: `Faithfulness reflects how accurately the explanation corresponds to the model's true reasoning, while plausibility refers to how convincing or understandable it is to a human. A plausible but unfaithful explanation may mislead users into trusting incorrect decisions. Distinguishing them is essential to avoid deceptive explanations and ensure reliable use of AI.`,
    topic: "XAI Evaluation"
  },
  {
    question: `What does mechanistic interpretability focus on?`,
    options: [
      "Explaining model predictions through concept-level visualizations",
      "Visualizing attention weights to track important tokens",
      "Understanding how internal components (e.g., neurons or attention heads) implement computations",
      "Analyzing correlations between inputs and predictions only"
    ],
    correct: "c",
    explanation: `Mechanistic interpretability aims to reverse-engineer how model components (like layers, neurons, circuits) carry out specific computations.`,
    topic: "Mechanistic Interpretability"
  },
  {
    question: `Which tool is often used to study the internal behavior of transformer models in mechanistic interpretability?`,
    options: [
      "Counterfactual explanations",
      "Concept whitening",
      "Probing methods and circuit analysis",
      "Partial dependence plots"
    ],
    correct: "c",
    explanation: `Mechanistic interpretability uses probing, ablation, and tracing to investigate circuits or neuron activations inside the model.`,
    topic: "Mechanistic Interpretability"
  },
  {
    question: `Why is mechanistic interpretability particularly relevant for large language models?`,
    options: [
      "Because they are easy to visualize globally",
      "Because their structure is purely interpretable by design",
      "Because their internal computation is opaque and needs fine-grained analysis",
      "Because they are trained without labeled data"
    ],
    correct: "c",
    explanation: `Due to their scale and complexity, LLMs are difficult to interpret globally. Mechanistic methods help isolate and analyze internal functions.`,
    topic: "Mechanistic Interpretability"
  },
  {
    question: `What is a "circuit" in the context of mechanistic interpretability?`,
    expected_answer: `A group of neurons or attention heads that work together to implement a specific function or behavior inside a model.`,
    topic: "Mechanistic Interpretability"
  },
  {
    question: `What is one major challenge of mechanistic interpretability?`,
    expected_answer: `It requires substantial manual effort and often lacks generalization across architectures or tasks.`,
    topic: "Mechanistic Interpretability"
  },
  {
    question: `Compare mechanistic interpretability to post-hoc explanation methods like LIME or SHAP. When is each approach more suitable?`,
    expected_answer: `Mechanistic interpretability looks inside the model to understand how its components work, offering deep insights but requiring model access. LIME/SHAP are black-box post-hoc methods useful when internals are unavailable. LIME is better for local, fast, model-agnostic explanations, while mechanistic analysis is better for scientific understanding of model internals.`,
    topic: "Mechanistic Interpretability"
  },
  {
    question: `What is the goal of an adversarial attack in machine learning?`,
    options: [
      "To optimize a model using transfer learning",
      "To evaluate model fairness on a benchmark",
      "To fool a model into making wrong predictions by adding small, imperceptible perturbations",
      "To improve training accuracy by removing noisy data"
    ],
    correct: "c",
    explanation: `Adversarial attacks aim to slightly modify inputs so that a model makes incorrect predictions, revealing vulnerabilities in the system.`,
    topic: "Adversarial Attacks and Counterfactual Explanations"
  },
  {
    question: `What is a key characteristic of counterfactual explanations?`,
    options: [
      "They provide global summaries of model behavior",
      "They generate explanations for all training instances simultaneously",
      "They identify minimal and plausible changes to the input that flip the model’s decision",
      "They use the gradient to compute token importance"
    ],
    correct: "c",
    explanation: `Counterfactual explanations describe how to minimally change an input to obtain a different outcome, which is useful for actionable insights.`,
    topic: "Adversarial Attacks and Counterfactual Explanations"
  },
  {
    question: `Which technique improves model robustness against adversarial attacks?`,
    options: [
      "Model ensembling",
      "Adversarial training with perturbed examples",
      "Removing outliers from the dataset",
      "Using PCA for dimensionality reduction"
    ],
    correct: "b",
    explanation: `Adversarial training includes adversarial examples during learning to enhance model robustness against perturbations.`,
    topic: "Adversarial Attacks and Counterfactual Explanations"
  },
  {
    question: `What is the key difference between adversarial attacks and counterfactual explanations?`,
    expected_answer: `Adversarial attacks aim to fool the model by generating inputs that cause wrong outputs, often maliciously. Counterfactual explanations aim to provide interpretable insights by describing how to obtain a desired output.`,
    topic: "Adversarial Attacks and Counterfactual Explanations"
  },
  {
    question: `Why are counterfactual explanations considered actionable?`,
    expected_answer: `Because they describe what specific feature changes a user could make to receive a different outcome from the model, such as increasing income to get loan approval.`,
    topic: "Adversarial Attacks and Counterfactual Explanations"
  },
  {
    question: `Describe a real-world scenario where counterfactual explanations and robustness to adversarial attacks are both important. What techniques can be used to support both interpretability and security?`,
    expected_answer: `In medical AI systems, counterfactuals help explain what changes in patient data lead to different diagnoses. Simultaneously, robustness is crucial to prevent errors from malicious inputs. Combining interpretable models with adversarial training and formal certification helps ensure both transparency and reliability.`,
    topic: "Adversarial Attacks and Counterfactual Explanations"
  }
];
