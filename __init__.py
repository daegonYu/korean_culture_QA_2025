import json
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm
import re
import os
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from rouge_score import rouge_scorer
import argparse
import warnings
from peft import PeftConfig
from dotenv import load_dotenv
import wandb

warnings.filterwarnings('ignore')