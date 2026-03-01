"""GQA dataset handler - Visual Question Answering."""
import os
import re
from typing import Dict, List, Optional

from utils.reward_score import gqa as gqa_reward

from .base import DatasetHandler

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Common GQA synonyms: maps alternative words -> canonical form
_SYNONYMS = {
    "sofa": "couch", "automobile": "car", "auto": "car",
    "phone": "telephone", "cellphone": "telephone", "cell phone": "telephone",
    "tv": "television", "bike": "bicycle", "motorbike": "motorcycle",
    "aeroplane": "airplane", "plane": "airplane",
    "kid": "child", "kids": "children",
    "road": "street", "sidewalk": "pavement",
    "mom": "mother", "dad": "father",
    "huge": "large", "big": "large", "tiny": "small", "little": "small",
    # Window coverings (very common GQA confusion)
    "curtain": "drape", "drape": "drape",
    # Common VQA synonyms
    "lady": "woman", "guy": "man",
    "laptop": "computer", "pc": "computer",
    "puppy": "dog", "pup": "dog",
    "kitten": "cat", "kitty": "cat",
}

# Default image directory (relative to project root)
_DEFAULT_IMAGE_DIR = "data/gqa/images"


class GQAHandler(DatasetHandler):
    """Handler for the GQA visual question answering dataset."""

    name = "gqa"
    default_train_path = "data/gqa/train.parquet"
    default_test_path = "data/gqa/testdev.parquet"
    default_max_tokens = 256
    # Flag: whether loaded data has images (set during load_data)
    has_images = true

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def load_data(
        self,
        path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict]:
        """Load GQA dataset from parquet files, with optional image support.
        
        If images exist in data/gqa/images/{imageId}.jpg, they will be loaded
        and included in the prompts for VL models. Otherwise falls back to text-only.
        
        To prepare data with images:
            sbatch --export=ALL,DATASETS=gqa scripts/prepare_data.sh
        """
        import pandas as pd
        
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data not found: {path}. Run: sbatch --export=ALL,DATASETS=gqa scripts/prepare_data.sh"
            )
        
        df = pd.read_parquet(path)
        
        # Check if images directory exists (data/gqa/images/)
        img_dir = os.path.join(os.path.dirname(path), "images")
        if not os.path.isdir(img_dir):
            img_dir = _DEFAULT_IMAGE_DIR
        
        # Probe: check multiple samples to confirm images are actually there
        probe_ids = [str(row.get("imageId", "")) for _, row in df.head(5).iterrows()]
        probe_found = sum(1 for pid in probe_ids if pid and os.path.exists(os.path.join(img_dir, f"{pid}.jpg")))
        images_available = probe_found >= 1
        
        task_datas = []
        with_images = 0
        missing_images = 0
        for idx, row in df.iterrows():
            if idx < start_index:
                continue
            
            question = row["question"]
            answer = str(row["answer"]).strip().lower()
            full_answer = row.get("fullAnswer", "")
            image_id = str(row.get("imageId", ""))
            
            text_prompt = (
                f"Look at the image and answer the question.\n\n"
                f"Question: {question}\n\n"
                f"Please reason step by step, and put your final answer within \\boxed{{}}."
            )
            
            # Build messages: multimodal if image exists, text-only otherwise
            image_path = os.path.join(img_dir, f"{image_id}.jpg") if image_id else ""
            if images_available and image_path and os.path.exists(image_path):
                # Multimodal message format (compatible with Qwen2.5-VL, LLaVA, etc.)
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text_prompt},
                ]}]
                data_entry = {
                    "messages": messages,
                    "image_path": image_path,
                }
                with_images += 1
            else:
                if images_available:
                    missing_images += 1
                messages = [{"role": "user", "content": text_prompt}]
                data_entry = {
                    "messages": messages,
                }
            
            data_entry.update({
                "ground_truth": {
                    "answer": answer,
                    "full_answer": str(full_answer) if full_answer else "",
                },
                "question_id": row.get("id", idx),
            })
            task_datas.append(data_entry)
            if max_samples and len(task_datas) >= max_samples:
                break
        
        self.has_images = images_available
        n = len(task_datas)
        if images_available:
            print(f"  [GQA] {split}: {with_images}/{n} samples have images, {missing_images} missing → {'multimodal' if with_images > 0 else 'text-only'} mode")
            if missing_images > 0:
                print(f"  [GQA] To fix missing images: sbatch --export=ALL,DATASETS=gqa scripts/prepare_data.sh")
        else:
            print(f"  [GQA] {split}: no images in {img_dir}/ → text-only mode")
            print(f"  [GQA] To download images: sbatch --export=ALL,DATASETS=gqa scripts/prepare_data.sh")
        
        return task_datas

    # -------------------------------------------------------------------------
    # Normalization helpers
    # -------------------------------------------------------------------------
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison: lowercase, strip, remove articles/punctuation."""
        text = text.strip().lower()
        # Remove leading/trailing punctuation
        text = re.sub(r'^[.!?,;:\'"()\-]+', '', text)
        text = re.sub(r'[.!?,;:\'"()\-]+$', '', text)
        # Remove leading articles
        text = re.sub(r'^(a|an|the)\s+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def _singularize(word: str) -> str:
        """Basic singular form: women->woman, buses->bus, dogs->dog, etc.
        
        Preserves common short words that end in 's' but are NOT plurals:
        yes, no, this, his, was, has, is, bus, gas, etc.
        """
        w = word.lower()
        # Words ending in 's' that should NOT be singularized
        _NO_STRIP = {
            "yes", "no", "this", "his", "its", "is", "was", "has", "does",
            "us", "bus", "gas", "plus", "minus", "lens", "canvas", "glass",
            "grass", "dress", "across", "always", "sometimes", "perhaps",
            "whereas", "less", "unless", "bonus", "campus", "focus", "status",
            "radius", "virus", "cactus", "fungus", "genius", "stimulus",
        }
        if w in _NO_STRIP:
            return w
        irregulars = {
            "women": "woman", "men": "man", "children": "child",
            "people": "person", "mice": "mouse", "feet": "foot",
            "teeth": "tooth", "geese": "goose", "oxen": "ox",
        }
        if w in irregulars:
            return irregulars[w]
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("ves"):
            return w[:-3] + "f"
        if w.endswith("ses") or w.endswith("xes") or w.endswith("zes") or w.endswith("ches") or w.endswith("shes"):
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
            return w[:-1]
        return w
    
    @staticmethod
    def _canonicalize(word: str) -> str:
        """Map synonyms to canonical form."""
        w = word.lower()
        return _SYNONYMS.get(w, w)

    # -------------------------------------------------------------------------
    # Matching
    # -------------------------------------------------------------------------
    
    def _match_words(self, pred: str, gt: str) -> bool:
        """Match two normalized answer strings flexibly (exact, plural, synonym)."""
        if not pred or not gt:
            return False
        if pred == gt:
            return True
        # Singular form match
        if self._singularize(pred) == self._singularize(gt):
            return True
        # Synonym match
        if self._canonicalize(pred) == self._canonicalize(gt):
            return True
        if self._canonicalize(self._singularize(pred)) == self._canonicalize(self._singularize(gt)):
            return True
        return False
    
    def _match_answer(self, pred: str, gt: str) -> bool:
        """Flexible answer matching: exact, containment, plural, synonym."""
        if not pred or not gt:
            return False
        # Direct match (handles single and multi-word)
        if self._match_words(pred, gt):
            return True
        # Containment: gt in pred or pred in gt
        if gt in pred or pred in gt:
            return True
        # Multi-word: word-by-word singular/synonym comparison
        pred_words = pred.split()
        gt_words = gt.split()
        if len(pred_words) == len(gt_words) and len(pred_words) <= 3:
            if all(self._match_words(p, g) for p, g in zip(pred_words, gt_words)):
                return True
        return False
    
    def _whole_word_search(self, text: str, gt: str) -> bool:
        """Check if gt appears as a whole word (or its singular/synonym) in text.
        
        This is the flexible fallback: scan the entire response for the answer.
        Safe for GQA because answers are short (1-2 words).
        """
        if not text or not gt:
            return False
        text_lower = text.lower()
        gt_lower = gt.lower()
        
        # Direct whole-word search
        if re.search(r'\b' + re.escape(gt_lower) + r'\b', text_lower):
            return True
        
        # Search for singular form of gt
        gt_sing = self._singularize(gt_lower)
        if gt_sing != gt_lower and re.search(r'\b' + re.escape(gt_sing) + r'\b', text_lower):
            return True
        
        # Search for synonym of gt
        gt_canon = self._canonicalize(gt_lower)
        if gt_canon != gt_lower and re.search(r'\b' + re.escape(gt_canon) + r'\b', text_lower):
            return True
        
        # For multi-word gt: check if all words appear in text
        gt_words = gt_lower.split()
        if len(gt_words) >= 2:
            if all(re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in gt_words):
                return True
            # Also try singular forms
            if all(re.search(r'\b' + re.escape(self._singularize(w)) + r'\b', text_lower) for w in gt_words):
                return True
        
        # Scan every 1-2 word window in text for singular/synonym match
        text_words = re.findall(r'\b\w+\b', text_lower)
        for w in text_words:
            if self._match_words(w, gt_lower):
                return True
        # 2-word windows
        if len(gt_words) == 2:
            for i in range(len(text_words) - 1):
                bigram = text_words[i] + " " + text_words[i + 1]
                if self._match_words(bigram, gt_lower):
                    return True
        
        return False

    # -------------------------------------------------------------------------
    # Answer extraction
    # -------------------------------------------------------------------------
    
    def _extract_boxed(self, response: str) -> Optional[str]:
        return gqa_reward.extract_boxed(response)
    
    def extract_answer(self, response: str) -> str:
        """Extract short answer from model response."""
        return gqa_reward.extract_answer(response)
    
    def _clean_extracted(self, text: str) -> Optional[str]:
        """Clean extracted text from regex match, return None if too long."""
        return gqa_reward.clean_extracted(text)

    # -------------------------------------------------------------------------
    # Reward and correctness
    # -------------------------------------------------------------------------
    
    def compute_reward(self, response: str, ground_truth) -> float:
        """Reward with \\boxed{} format preference.
        
        - 1.0 if answer extracted from \\boxed{} matches ground truth
        - 1.0 if answer extracted via fallback patterns matches ground truth
        - 1.0 if whole-response scan finds ground truth (last resort)
        - 0.0 otherwise
        """
        return gqa_reward.compute_score(
            response=response,
            ground_truth=ground_truth,
            extract_answer=self.extract_answer,
            normalize_answer=self._normalize_answer,
            match_answer=self._match_answer,
            whole_word_search=self._whole_word_search,
        )
    
    def is_answer_correct(self, response: str, ground_truth) -> bool:
        """Check if answer is correct (flexible)."""
        return self.compute_reward(response, ground_truth) > 0
    
    def is_voted_answer_correct(self, voted_answer: str, ground_truth) -> bool:
        """Check if voted answer matches ground truth (flexible).
        
        The voted_answer comes from extract_answer_for_voting which already
        applies singularize + canonicalize. So we apply the same to GT.
        Uses both _match_answer (exact/containment) and _whole_word_search
        (scan for GT word in pred) to match the flexibility of compute_reward.
        """
        if isinstance(ground_truth, dict):
            gt_answer = ground_truth.get("answer", "")
        else:
            gt_answer = str(ground_truth)
        
        pred = self._normalize_answer(voted_answer)
        gt = self._normalize_answer(gt_answer)
        
        # Also try canonicalized GT for fair comparison with canonicalized pred
        gt_words = gt.split()
        gt_canonical = ' '.join(self._canonicalize(self._singularize(w)) for w in gt_words)
        
        # Primary: exact/containment match
        if self._match_answer(pred, gt) or self._match_answer(pred, gt_canonical):
            return True
        
        # Fallback: whole-word scan (same flexibility as compute_reward)
        # This catches cases like pred="curtain and small table" with gt="drape"
        if self._whole_word_search(pred, gt):
            return True
        
        return False

    def extract_answer_for_voting(self, response: str) -> str:
        """Extract normalized answer for voting (ensures consistent format).
        
        Applies synonym canonicalization + singularization so that
        equivalent answers (e.g. 'curtains' vs 'drapes', 'kids' vs 'children')
        map to the same vote key.
        """
        answer = self.extract_answer(response)
        normalized = self._normalize_answer(answer)
        if not normalized:
            return ""
        # Canonicalize each word: singularize + synonym mapping
        words = normalized.split()
        canonical_words = [self._canonicalize(self._singularize(w)) for w in words]
        return ' '.join(canonical_words)

    def format_answer_for_check(self, answer: str) -> str:
        """Format answer in \\boxed{} for display/checking."""
        return f"\\boxed{{{answer}}}"

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------
    
    def postprocess_outputs_with_debug(self, outputs, task_datas, sample_size=20):
        """Debug: show sample outputs with extraction details."""
        import random
        indices = list(range(len(outputs)))
        random.shuffle(indices)
        
        correct, incorrect, total = 0, 0, len(outputs)
        incorrect_examples = []
        
        for i, (output, data) in enumerate(zip(outputs, task_datas)):
            response = output.outputs[0].text
            gt = data.get("ground_truth", {})
            gt_answer = gt.get("answer", "") if isinstance(gt, dict) else str(gt)
            reward = self.compute_reward(response, gt)
            if reward > 0:
                correct += 1
            else:
                incorrect += 1
                incorrect_examples.append((i, response, gt_answer))
        
        return {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": correct / total if total > 0 else 0,
            "incorrect_examples": incorrect_examples[:sample_size],
        }
    
    def print_debug_report(self, debug_info):
        """Print debug report showing failed extractions."""
        print(f"\n{'='*60}")
        print(f"GQA Debug Report: {debug_info['correct']}/{debug_info['total']} correct ({debug_info['accuracy']*100:.1f}%)")
        print(f"{'='*60}")
        print(f"\nSample incorrect predictions ({len(debug_info['incorrect_examples'])} shown):")
        for idx, response, gt in debug_info['incorrect_examples']:
            pred = self._normalize_answer(self.extract_answer(response))
            print(f"  [{idx:4d}] GT: '{gt}' | Extracted: '{pred}' | Raw: '{response[:80]}'")
        print()

