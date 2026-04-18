# Tenne: AI-Generated Content Detection Model
## 📖 Product Overview
Tenne is a binary classification model specifically designed to distinguish between **human-written** and **AI-generated** text. The model effectively identifies content produced by large language models such as ChatGPT, providing technical support for content moderation, academic integrity detection, and related applications.
## ✅ Core Capabilities
- High Accuracy: 97%+ accuracy on validation set    
- Zero False Negatives: 100% recall for AI-generated text    
- Chinese Optimized: Built on BERT-base-chinese architecture    
- Fast Inference: Millisecond-level response time    
## 🎯 Performance Metrics
### Overall Metrics 
| Metric | Score |
| ------ | ----- |
|Accuracy|97.17% |
|F1 Score|0.965  |
|AUC     |1.000  |

### Classification Performance

| Category | Precision | Recall | 
| -------- | --------- | ------ |
|Human Text|   100%	   | 95.35% |
|AI Text   |   93.26%  |100%    |

## 🚀 Quick Start

### Requirements


```bash
pip install torch transformers
```

### Load Model from Hugging Face

```python

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import torch

tokenizer = AutoTokenizer.from_pretrained("MAHE-model/Tenne")
model = AutoModelForSequenceClassification.from_pretrained("MAHE-model/Tenne")
model.eval()
```

### Single Text Detection


```python
def detect(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    
    label = "AI Generated" if probs[0][1] > 0.5 else "Human Written"
    confidence = probs[0][1].item() if label == "AI Generated" else probs[0][0].item()
    
    return {
        "label": label, 
        "confidence": confidence, 
        "ai_probability": probs[0][1].item()
    }

# Example usage
result = detect("Text content to be analyzed")
print(f"Result: {result['label']}, Confidence: {result['confidence']:.4f}")
```

### Batch Detection


``` python
def batch_detect(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512, 
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        for j, prob in enumerate(probs):
            results.append({
                "text": batch[j],
                "label": "AI Generated" if prob[1] > 0.5 else "Human Written",
                "ai_score": prob[1].item()
            })
    return results
```

## 💼 Use Cases

| Scenario        |     Description |
| --------------- | --------------- |
| Academic Integrity | Detect AI-generated student essays and papers |
| Content Moderation | Identify AI-generated content on platforms |
| Recruitment Screening | Evaluate authenticity of resumes and cover letters |
| News Verification | Determine if news articles are AI-generated |

## 📊 Model Specifications

| Parameter | Specification |
|---------- | ------------- |
| Model Size | ~400 MB |
| Max Input Length | 512 tokens |
| Inference Time | < 100ms (GPU) |
| Language | Chinese (Simplified) |

## 📞 Business Inquiries
For model files, API keys, or enterprise deployment solutions, please contact:
Website: https://maherjon.github.io/MAHE/

## 🔒 Copyright Notice
This model and related technologies are proprietary. Commercial use without authorization is prohibited.
