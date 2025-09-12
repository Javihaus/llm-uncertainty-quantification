"""
Dataset loading and preprocessing for uncertainty quantification experiments.
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset, Dataset
import random
import os


class TruthfulQAProcessor:
    """Process TruthfulQA dataset for uncertainty evaluation."""
    
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        self.dataset = None
    
    def load_dataset(self) -> Dataset:
        """Load TruthfulQA dataset."""
        try:
            # Load the multiple choice version
            dataset = load_dataset("truthful_qa", "multiple_choice")
            self.dataset = dataset['validation']
            print(f"Loaded TruthfulQA with {len(self.dataset)} samples")
            
            # Sample subset if specified
            if self.num_samples and self.num_samples < len(self.dataset):
                indices = random.sample(range(len(self.dataset)), self.num_samples)
                self.dataset = self.dataset.select(indices)
                print(f"Sampled {self.num_samples} examples")
            
            return self.dataset
            
        except Exception as e:
            print(f"Error loading TruthfulQA: {e}")
            # Create dummy dataset for testing
            return self._create_dummy_dataset()
    
    def _create_dummy_dataset(self) -> List[Dict]:
        """Create dummy TruthfulQA-like dataset for testing."""
        dummy_data = [
            {
                "question": "What is the capital of France?",
                "mc1_targets": {"choices": ["Paris", "London", "Berlin", "Madrid"], "labels": [1, 0, 0, 0]},
                "mc2_targets": {"choices": ["Paris", "London", "Berlin", "Madrid"], "labels": [1, 0, 0, 0]}
            },
            {
                "question": "Is the Earth flat?",
                "mc1_targets": {"choices": ["No, the Earth is round", "Yes, the Earth is flat"], "labels": [1, 0]},
                "mc2_targets": {"choices": ["No, the Earth is round", "Yes, the Earth is flat"], "labels": [1, 0]}
            },
            {
                "question": "What happens if you break a mirror?",
                "mc1_targets": {"choices": ["You have a broken mirror", "You get seven years of bad luck"], "labels": [1, 0]},
                "mc2_targets": {"choices": ["You have a broken mirror", "You get seven years of bad luck"], "labels": [1, 0]}
            }
        ] * (self.num_samples // 3 + 1)
        
        return dummy_data[:self.num_samples]
    
    def format_for_evaluation(self) -> List[Dict]:
        """Format dataset for uncertainty evaluation."""
        formatted_data = []
        
        if self.dataset is None:
            self.load_dataset()
        
        # Handle both real and dummy datasets
        if hasattr(self.dataset, '__len__') and hasattr(self.dataset, '__getitem__'):
            # Real dataset
            for item in self.dataset:
                formatted_data.append(self._format_item(item))
        else:
            # Dummy dataset
            for item in self.dataset:
                formatted_data.append(self._format_item(item))
        
        return formatted_data
    
    def _format_item(self, item: Dict) -> Dict:
        """Format single item for evaluation."""
        question = item['question']
        choices = item['mc1_targets']['choices']
        labels = item['mc1_targets']['labels']
        
        # Find correct answer
        correct_idx = labels.index(1) if 1 in labels else 0
        correct_answer = choices[correct_idx]
        
        return {
            'question': question,
            'choices': choices,
            'correct_answer': correct_answer,
            'correct_idx': correct_idx,
            'prompt': f"Question: {question}\nAnswer:",
            'type': 'multiple_choice'
        }


class MMLUProcessor:
    """Process MMLU dataset subsets for uncertainty evaluation."""
    
    def __init__(self, subjects: List[str] = None, num_samples_per_subject: int = 50):
        self.subjects = subjects or ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics']
        self.num_samples_per_subject = num_samples_per_subject
        self.dataset = None
    
    def load_dataset(self) -> List[Dict]:
        """Load MMLU dataset subsets."""
        all_data = []
        
        for subject in self.subjects:
            try:
                dataset = load_dataset("cais/mmlu", subject)
                test_data = dataset['test']
                
                # Sample subset
                if self.num_samples_per_subject < len(test_data):
                    indices = random.sample(range(len(test_data)), self.num_samples_per_subject)
                    test_data = test_data.select(indices)
                
                for item in test_data:
                    formatted_item = self._format_mmlu_item(item, subject)
                    all_data.append(formatted_item)
                    
                print(f"Loaded {len(test_data)} samples from {subject}")
                
            except Exception as e:
                print(f"Error loading MMLU {subject}: {e}")
                # Add dummy data
                all_data.extend(self._create_dummy_mmlu_data(subject))
        
        self.dataset = all_data
        print(f"Total MMLU samples: {len(all_data)}")
        return all_data
    
    def _format_mmlu_item(self, item: Dict, subject: str) -> Dict:
        """Format MMLU item for evaluation."""
        question = item['question']
        choices = item['choices']
        correct_idx = item['answer']
        correct_answer = choices[correct_idx]
        
        # Create formatted prompt
        choices_text = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        prompt = f"Question: {question}\n{choices_text}\nAnswer:"
        
        return {
            'question': question,
            'choices': choices,
            'correct_answer': correct_answer,
            'correct_idx': correct_idx,
            'prompt': prompt,
            'subject': subject,
            'type': 'multiple_choice'
        }
    
    def _create_dummy_mmlu_data(self, subject: str) -> List[Dict]:
        """Create dummy MMLU data for testing."""
        dummy_questions = [
            {
                'question': f'What is a basic concept in {subject}?',
                'choices': ['Option A', 'Option B', 'Option C', 'Option D'],
                'answer': 0
            }
        ] * self.num_samples_per_subject
        
        return [self._format_mmlu_item(q, subject) for q in dummy_questions]


class FactualKnowledgeProcessor:
    """Process simple factual knowledge questions."""
    
    def __init__(self, num_samples: int = 50):
        self.num_samples = num_samples
    
    def create_dataset(self) -> List[Dict]:
        """Create factual knowledge dataset."""
        factual_qa = [
            {
                'question': 'What is the capital of the United States?',
                'correct_answer': 'Washington D.C.',
                'type': 'open_ended'
            },
            {
                'question': 'Who wrote Romeo and Juliet?',
                'correct_answer': 'William Shakespeare',
                'type': 'open_ended'
            },
            {
                'question': 'What is 2 + 2?',
                'correct_answer': '4',
                'type': 'open_ended'
            },
            {
                'question': 'What is the largest planet in our solar system?',
                'correct_answer': 'Jupiter',
                'type': 'open_ended'
            },
            {
                'question': 'In what year did World War II end?',
                'correct_answer': '1945',
                'type': 'open_ended'
            },
            {
                'question': 'What is the chemical symbol for gold?',
                'correct_answer': 'Au',
                'type': 'open_ended'
            },
            {
                'question': 'Who painted the Mona Lisa?',
                'correct_answer': 'Leonardo da Vinci',
                'type': 'open_ended'
            },
            {
                'question': 'What is the speed of light?',
                'correct_answer': '299,792,458 meters per second',
                'type': 'open_ended'
            }
        ]
        
        # Extend dataset to desired size
        extended_data = (factual_qa * (self.num_samples // len(factual_qa) + 1))[:self.num_samples]
        
        # Add prompts
        for item in extended_data:
            item['prompt'] = f"Question: {item['question']}\nAnswer:"
        
        return extended_data


class DatasetManager:
    """Manage all datasets for experiments."""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.datasets = {}
    
    def load_all_datasets(self, 
                         truthful_qa_samples: int = 100,
                         mmlu_samples_per_subject: int = 50,
                         factual_samples: int = 50) -> Dict[str, List[Dict]]:
        """Load all evaluation datasets."""
        
        print("Loading TruthfulQA...")
        truthful_processor = TruthfulQAProcessor(truthful_qa_samples)
        self.datasets['truthfulqa'] = truthful_processor.format_for_evaluation()
        
        print("Loading MMLU subsets...")
        mmlu_processor = MMLUProcessor(num_samples_per_subject=mmlu_samples_per_subject)
        self.datasets['mmlu'] = mmlu_processor.load_dataset()
        
        print("Creating factual knowledge dataset...")
        factual_processor = FactualKnowledgeProcessor(factual_samples)
        self.datasets['factual'] = factual_processor.create_dataset()
        
        # Save datasets
        self._save_datasets()
        
        return self.datasets
    
    def _save_datasets(self):
        """Save datasets to disk."""
        for name, dataset in self.datasets.items():
            save_path = os.path.join(self.data_dir, f"{name}.json")
            with open(save_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            print(f"Saved {name} dataset to {save_path}")
    
    def load_saved_datasets(self) -> Dict[str, List[Dict]]:
        """Load previously saved datasets."""
        for name in ['truthfulqa', 'mmlu', 'factual']:
            file_path = os.path.join(self.data_dir, f"{name}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.datasets[name] = json.load(f)
                print(f"Loaded {name} dataset from {file_path}")
        
        return self.datasets
    
    def get_dataset_summary(self) -> Dict[str, Dict]:
        """Get summary statistics for all datasets."""
        summary = {}
        
        for name, dataset in self.datasets.items():
            summary[name] = {
                'num_samples': len(dataset),
                'types': list(set(item.get('type', 'unknown') for item in dataset)),
                'sample_question': dataset[0]['question'] if dataset else None
            }
            
            if name == 'mmlu':
                subjects = list(set(item.get('subject', 'unknown') for item in dataset))
                summary[name]['subjects'] = subjects
        
        return summary