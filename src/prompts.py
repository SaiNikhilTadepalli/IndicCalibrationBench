ACCURACY_GRADING_PROMPT = """
You are evaluating whether a submitted answer demonstrates an accurate and nuanced understanding of regional Indian culture, geography and heritage, specifically focusing on lesser-known locations in India.

Question: {question}

Correct answer: {correct_answer}

Submitted answer: {submitted_answer}

Evaluate whether the submitted answer is factually correct and demonstrates appropriate regional and thematic understanding. The submitted answer does not need to match the correct answer word-for-word, but it must convey the same specific factual details and regional context.

Consider the answer CORRECT if it:
- Contains the specific factual information from the correct answer (e.g., correct names of herbs, specific locations or mythological associations).
- Demonstrates accurate knowledge of the relevant relevant Indian regional context (e.g., specific states, communities or local practices).

Consider the answer INCORRECT if it:
- Contains factual errors regarding the specific Indian location or name.
- Confuses the fact with a different Indian region or uses incorrect terminology.
- Demonstrates a fundamental misunderstanding of the regional context or cultural significance.
- Completely misses the key unique identifiers and names provided in the reference answer. 

Provide your evaluation as either "CORRECT" or "INCORRECT" followed by a brief, one-sentence explanation of why the answer was graded this way.
"""

INDIC_CALIBRATION_SYSTEM_PROMPT = """
You are an expert on the regional diversity of India, covering lesser-known facts about food and culinary practices, trees and herbs, mythology, history and geography and other cultural themes.

For every question:
1. Provide a concise, factually accurate response that reflects specific local context.
2. Be extremely precise and accurate with regional names, species and cultural practices.
3. Critically assess your own certainty. You must provide a confidence score between 0 and 100, indicating how likely you believe your answer is to be correct.
"""

QUESTION_ANSWERING_TEMPLATE = """
Question: {question}

Please provide your answer followed by your confidence score (between 0 and 100) in the following format:

Answer: [Your answer]
Confidence: [Your confidence score]%
"""

CONFIDENCE_SCORE_REGEX = r"(?:Confidence:\s*)?(\d{1,3})\s*%"
