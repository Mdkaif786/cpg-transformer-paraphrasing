# Custom Paraphrase Generator

A specialized NLP system for generating high-quality paraphrases using transformer models. This project compares a custom paraphrase-specific model (T5_Paraphrase_Paws) against a general-purpose LLM (FLAN-T5-Large) using comprehensive evaluation metrics.

##  Problem Statement

Paraphrase generation is a critical NLP task with applications in content creation, text summarization, and data augmentation. The challenge lies in generating paraphrases that:
- Preserve the original meaning (semantic similarity)
- Maintain appropriate length (minimum 80% of input)
- Avoid hallucinations or irrelevant content
- Generate diverse yet accurate rephrasing

**Research Question:** Do specialized, smaller models outperform larger general-purpose LLMs for domain-specific tasks like paraphrasing?

##  Project Overview

This project implements a **Custom Paraphrase Generator (CPG)** that:
- Generates paraphrases for paragraphs (200-400 words input range)
- Ensures minimum output length of 80% of input length
- Compares specialized vs. general-purpose transformer models
- Evaluates using multiple metrics (ROUGE, BLEU, BERTScore)
- Analyzes performance, quality, and efficiency trade-offs

##  Key Results

| Metric | Custom Model (T5_Paraphrase_Paws) | LLM (FLAN-T5-Large) | Winner |
|--------|-----------------------------------|---------------------|--------|
| **Model Size** | 222M parameters | 783M parameters | Custom (3.5x smaller) |
| **Generation Time** | 13.73s | 16.52s | Custom (20% faster) |
| **Output Length** | 335 words (101.8%) | 275 words (83.6%) | Custom |
| **ROUGE-L** | 0.9422 | 0.6094 | Custom |
| **BLEU** | 0.6803 | 0.3964 | Custom |
| **BERTScore F1** | 0.9778 | 0.9130 | Custom |
| **Hallucinations** | None | Present | Custom |

**Key Finding:** The specialized 222M parameter model outperforms the 783M general-purpose model, demonstrating that **task-specific training matters more than model size** for domain-specific NLP tasks.

##  Technology Stack

### Models Used:
- **Custom Model:** `Vamsi/T5_Paraphrase_Paws` (222M parameters)
  - Fine-tuned on PAWS (Paraphrase Adversaries from Word Scrambling) dataset
  - T5 architecture optimized for paraphrase generation
  
- **Comparison Model:** `google/flan-t5-large` (783M parameters)
  - General-purpose instruction-tuned model
  - Baseline for comparison

### Libraries:
- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - Pre-trained model loading and inference
- **BERTScore** - Semantic similarity evaluation
- **ROUGE** - N-gram overlap metrics
- **BLEU** - Translation quality metrics
- **NLTK** - Natural language processing utilities
- **Matplotlib & Seaborn** - Data visualization

##  Environment Setup

### Prerequisites

- **Python:** 3.8 or higher
- **GPU:** CUDA-capable GPU recommended (for faster inference)
  - Minimum: 4GB VRAM
  - Recommended: 8GB+ VRAM
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** ~5GB free space (for model downloads)

### System Requirements

**For CPU-only execution:**
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 16GB+ recommended
- Note: Inference will be significantly slower (~5-10x)

**For GPU execution:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+ (if installing PyTorch with CUDA)
- cuDNN 8.0+ (for optimized performance)

##  Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/custom-paraphrase-generator.git
cd custom-paraphrase-generator
```

### Step 2: Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

**Option A: Using requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Manual Installation**
```bash
pip install torch transformers sentencepiece sacremoses evaluate rouge-score bert-score nltk textstat pandas numpy matplotlib seaborn
```

### Step 4: Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

Or run this in Python:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Step 5: Verify Installation

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

## 🚀 Usage

### Running the Project

#### Option 1: Jupyter Notebook (Recommended)

1. **Start Jupyter Notebook:**
```bash
jupyter notebook
```

2. **Open `Custom_Paraphraser.ipynb`**

3. **Run all cells sequentially:**
   - Cell 0: Install dependencies (if needed)
   - Cell 1: Import libraries and setup
   - Cell 2: Define test passage
   - Cell 3: Load custom model
   - Cell 4: Define paraphrase generation function
   - Cell 5: Generate paraphrase with custom model
   - Cell 6: Load comparison LLM
   - Cell 7: Generate paraphrase with LLM
   - Cell 8: Compare results
   - Cell 9: Calculate ROUGE and BLEU scores
   - Cell 10: Calculate BERTScore
   - Cell 11: Save results
   - Cell 12: Generate visualization

4. **Results will be saved:**
   - `results.json` - Evaluation metrics and outputs
   - `comparison_chart.png` - Visualization of results

#### Option 2: Google Colab

1. Upload `Custom_Paraphraser.ipynb` to Google Colab
2. Enable GPU runtime: Runtime → Change runtime type → GPU
3. Run all cells sequentially
4. Download results files

### Customizing Input Text

To paraphrase your own text, modify the `test_passage` variable in Cell 2:

```python
test_passage = """Your text here (200-400 words recommended)"""
```

**Requirements:**
- Input length: 200-400 words (recommended)
- Output will be at least 80% of input length
- Text should be well-formatted with proper punctuation

##  Evaluation Metrics

The system evaluates paraphrases using five key metrics:

### 1. ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L)
- **Purpose:** Measures n-gram overlap between original and generated text
- **Range:** 0-1 (higher = more overlap)
- **Interpretation:** 
  - Note: In paraphrasing tasks, very high ROUGE scores should be interpreted carefully. They must be analyzed alongside semantic metrics like BERTScore and qualitative inspection to ensure genuine rephrasing rather than surface-level similarity.
  - Too low (<0.3): May lose meaning
  - Moderate scores (0.6-0.9): Generally indicate good balance for paraphrasing

### 2. BLEU Score
- **Purpose:** Measures translation/paraphrase quality
- **Range:** 0-1 (higher is better)
- **Interpretation:** Similar to ROUGE, measures quality of rephrasing

### 3. BERTScore
- **Purpose:** Measures semantic similarity using contextual embeddings
- **Range:** 0-1 (higher = better meaning preservation)
- **Interpretation:** Most important metric for paraphrase quality
  - >0.95: Excellent semantic preservation
  - 0.90-0.95: Good preservation
  - <0.90: May lose some meaning

### 4. Length Ratio
- **Purpose:** Ensures output meets minimum length requirement
- **Requirement:** Output ≥ 80% of input length
- **Calculation:** (Output words / Input words) × 100%

### 5. Generation Time
- **Purpose:** Measures inference speed
- **Unit:** Seconds
- **Factors:** Model size, hardware (GPU/CPU), input length

## 🔬 Methodology

### Custom Model Approach

The paraphrase generation follows a sentence-by-sentence processing pipeline:

1. **Text Segmentation:**
   - Split input into individual sentences using regex
   - Handle edge cases (very short sentences < 3 words)

2. **Multi-Candidate Generation:**
   - Generate 4 candidate paraphrases per sentence
   - Use beam search with diversity sampling

3. **Candidate Selection:**
   - Calculate similarity between original and each candidate
   - Select candidate with optimal similarity (15-50% range)
   - Ensures meaningful paraphrasing without losing meaning

4. **Quality Filtering:**
   - Filter candidates based on similarity thresholds
   - Ensure minimum length preservation

5. **Post-Processing:**
   - Combine sentences with proper spacing
   - Clean up formatting issues

### Generation Parameters

```python
{
    'num_return_sequences': 4,      # Multiple candidates per sentence
    'num_beams': 4,                 # Beam search width
    'temperature': 0.9,             # Diversity control
    'top_k': 120,                   # Vocabulary restriction
    'top_p': 0.95,                  # Nucleus sampling
    'max_length': 128,              # Maximum tokens per sentence
    'early_stopping': True          # Stop when EOS token generated
}
```

### Test Sample

The project uses a **329-word cover letter passage** as the test sample:
- **Input:** 329 words (within 200-400 word range)
- **Minimum Required Output:** 263 words (80% of input)
- **Content Type:** Formal, structured text with multiple sections

##  Project Structure

```
custom_paraphrase_generator/
├── Custom_Paraphraser.ipynb    # Main Jupyter notebook (source code)
├── results.json                 # Evaluation results and outputs
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── comparison_chart.png        # Visualization (optional)
```

##  Key Insights

1. **Specialization Over Scale:** 
   - Smaller specialized model (222M) outperforms larger general model (783M)
   - Task-specific training is more valuable than raw parameter count

2. **Efficiency Gains:**
   - Custom model is 20% faster despite processing more content
   - Better resource utilization and lower computational cost

3. **Quality Assurance:**
   - Custom model achieves 97.78 BERTScore (excellent semantic preservation)
   - No observable hallucinations in the evaluated sample vs. LLM's hallucination issues

4. **Reliability:**
   - Custom model produces consistent, high-quality output
   - LLM shows tendency to add irrelevant content and incomplete sentences

##  Limitations

- Evaluation performed on a single long-form passage (329 words)
- No human evaluation conducted
- Performance may vary across domains (technical, creative, informal)
- Generation parameters tuned for this experiment

## 🔍 Results Analysis

### Custom Model Strengths:
-  Preserves all key information
-  Maintains formal tone and structure
-  Proper sentence structure and grammar
-  No observable hallucinations or irrelevant content in evaluated sample
-  Good word substitutions while maintaining meaning

### LLM Weaknesses:
-  Adds hallucinations (e.g., "USATODAY.com", "wikihow")
-  Incomplete sentences
-  Meaning loss in some sections
-  Repetitive/off-topic content

## 🐛 Troubleshooting

### Common Issues:

**1. CUDA Out of Memory Error**
```python
# Solution: Use CPU or reduce batch size
device = torch.device('cpu')
# Or reduce max_length in generation parameters
```

**2. Model Download Fails**
```python
# Solution: Set Hugging Face token for authenticated access
from huggingface_hub import login
login(token="your_token_here")
```

**3. NLTK Data Not Found**
```python
# Solution: Manually download
import nltk
nltk.download('punkt', download_dir='/path/to/nltk_data')
nltk.download('punkt_tab', download_dir='/path/to/nltk_data')
```

**4. Slow Inference on CPU**
- Expected behavior: CPU is 5-10x slower than GPU
- Consider using Google Colab with free GPU
- Or reduce input length for faster testing

##  License

This project is for educational and research purposes.

##  Acknowledgments

- **Models:** Hugging Face Transformers
  - `Vamsi/T5_Paraphrase_Paws`
  - `google/flan-t5-large`
  
- **Evaluation Metrics:**
  - ROUGE: Lin, C.Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries
  - BLEU: Papineni, K. et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation
  - BERTScore: Zhang, T. et al. (2020). BERTScore: Evaluating Text Generation with BERT

##  Future Work

-  Human evaluation study
-  Domain generalization experiments
-  Fine-tuning FLAN-T5 on paraphrase-specific data
-  Testing with Pegasus and BART models
-  Deployment as REST API

##  References

1. T5: Text-to-Text Transfer Transformer (Raffel et al., 2020)
2. PAWS: Paraphrase Adversaries from Word Scrambling (Zhang et al., 2019)
3. FLAN-T5: Instruction Tuning for Zero-Shot Learning (Chung et al., 2022)

---

**Note:** This project demonstrates that task-specific model specialization can outperform larger general-purpose models, providing valuable insights for NLP model selection and deployment in production environments.
