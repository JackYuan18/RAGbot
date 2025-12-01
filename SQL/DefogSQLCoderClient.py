#!/usr/bin/env python3
"""
Defog SQLCoder Client
Handles SQL generation using Defog's SQLCoder via HuggingFace transformers
"""

import logging
import os
from pathlib import Path
from typing import Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Try to import CodeLlama tokenizer classes in case they're needed
try:
    from transformers import CodeLlamaTokenizer, CodeLlamaTokenizerFast
except ImportError:
    # CodeLlama tokenizers might not be available in all transformers versions
    # AutoTokenizer should handle this automatically
    pass

# Set HuggingFace cache location to use shared cache
# This ensures all Python environments use the same cache
if 'HF_HOME' not in os.environ:
    hf_cache = os.path.expanduser('~/.cache/huggingface')
    os.environ['HF_HOME'] = hf_cache
    logging.getLogger(__name__).info(f"Set HF_HOME to: {hf_cache}")

class DefogSQLCoderClient:
    """Client for interacting with Defog's SQLCoder via HuggingFace transformers."""
    
    def __init__(self, model_name: str = "defog/sqlcoder-7b-2", prompt_file: Optional[str] = None):
        """
        Initialize Defog SQLCoder client.
        
        Args:
            model_name: HuggingFace model name (default: "defog/sqlcoder-7b-2")
            prompt_file: Path to prompt template file (default: uses defog_ai_sqlcoder/prompt.md)
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.model_loaded = False
        
        # Set up prompt file path
        if prompt_file is None:
            # Default to defog_ai_sqlcoder/prompt.md
            defog_dir = Path(__file__).parent / "defog_ai_sqlcoder"
            self.prompt_file = defog_dir / "prompt.md"
        else:
            self.prompt_file = Path(prompt_file)
        
        if not self.prompt_file.exists():
            self.logger.warning(f"Prompt file not found: {self.prompt_file}, using default prompt")
            self.prompt_file = None
    
    def _load_model(self):
        """Load the model and tokenizer if not already loaded."""
        if self.model_loaded:
            return
        
        try:
            self.logger.info(f"Loading Defog SQLCoder model: {self.model_name}")
            
            # Check cache location for debugging
            cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            hub_cache = os.path.join(cache_dir, "hub")
            model_cache_path = os.path.join(hub_cache, f"models--{self.model_name.replace('/', '--')}")
            self.logger.info(f"Model cache location: {hub_cache}")
            self.logger.info(f"Model cache path: {model_cache_path}")
            if os.path.exists(model_cache_path):
                self.logger.info(f"Model cache directory exists: {model_cache_path}")
            else:
                self.logger.warning(f"Model cache directory not found: {model_cache_path}")
            
            # Check for offline mode environment variable
            if os.environ.get("HF_HUB_OFFLINE", "").lower() == "1":
                raise ConnectionError(
                    "HuggingFace Hub is in offline mode. Set HF_HUB_OFFLINE=0 or unset it to download models."
                )
            
            # Try to load the model
            # ALWAYS try cached first with local_files_only=True
            # This ensures we use the cache if available, even without internet
            try:
                # Find the actual snapshot directory in cache
                cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                hub_cache = os.path.join(cache_dir, "hub")
                model_cache_path = os.path.join(hub_cache, f"models--{self.model_name.replace('/', '--')}")
                snapshot_dir = None
                
                if os.path.exists(model_cache_path):
                    # Look for snapshot directory
                    import glob
                    snapshots = glob.glob(os.path.join(model_cache_path, "snapshots", "*"))
                    if snapshots:
                        snapshot_dir = snapshots[0]  # Use the first snapshot
                        self.logger.info(f"Found snapshot directory: {snapshot_dir}")
                        # Verify config.json exists
                        config_path = os.path.join(snapshot_dir, "config.json")
                        if os.path.exists(config_path):
                            self.logger.info(f"Config.json found at: {config_path}")
                        else:
                            self.logger.warning(f"Config.json not found at: {config_path}")
                
                # Load tokenizer - prioritize snapshot directory if it exists
                if snapshot_dir and os.path.exists(snapshot_dir):
                    # Use snapshot directory directly (more reliable)
                    self.logger.info(f"Loading tokenizer from snapshot directory: {snapshot_dir}")
                    try:
                        # Try fast tokenizer first (doesn't need sentencepiece)
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            snapshot_dir,
                            trust_remote_code=True,
                            local_files_only=True,
                            use_fast=True  # Use fast tokenizer (doesn't need sentencepiece)
                        )
                        self.logger.info("Tokenizer loaded from snapshot directory (fast tokenizer)")
                    except Exception as snapshot_error:
                        # If snapshot loading with local_files_only fails, try without it
                        self.logger.warning(f"Failed to load from snapshot with local_files_only: {snapshot_error}")
                        self.logger.info("Trying to load tokenizer without local_files_only restriction...")
                        try:
                            # Try fast tokenizer first (doesn't need sentencepiece)
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                snapshot_dir,
                                trust_remote_code=True,
                                local_files_only=False,  # Allow downloading missing files
                                use_fast=True  # Use fast tokenizer
                            )
                            self.logger.info("Tokenizer loaded (fast tokenizer, may have downloaded missing files)")
                        except Exception as fast_error:
                            # If fast tokenizer fails, try slow tokenizer as last resort
                            self.logger.warning(f"Fast tokenizer failed: {fast_error}")
                            self.logger.info("Trying slow tokenizer (requires sentencepiece library)...")
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                snapshot_dir,
                                trust_remote_code=True,
                                local_files_only=False,  # Allow downloading missing files
                                use_fast=False  # Use slow tokenizer
                            )
                            self.logger.info("Tokenizer loaded (slow tokenizer)")
                else:
                    # No snapshot directory, try loading with model name
                    self.logger.info(f"Loading tokenizer for {self.model_name} from cache...")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            local_files_only=True  # Use cache only
                        )
                        self.logger.info("Tokenizer loaded from cache")
                    except Exception as e:
                        self.logger.warning(f"Failed to load tokenizer with model name: {str(e)}")
                        # If that fails, try without local_files_only
                        self.logger.info("Trying to load tokenizer without local_files_only restriction...")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            local_files_only=False
                        )
                        self.logger.info("Tokenizer loaded with model name")
                
                # Load model from cache
                # If we have a snapshot directory, use it directly
                if snapshot_dir and os.path.exists(snapshot_dir):
                    self.logger.info(f"Loading model from snapshot directory: {snapshot_dir}")
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            snapshot_dir,
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            low_cpu_mem_usage=True,  # Optimize memory usage with accelerate
                            use_cache=True,
                            local_files_only=True
                        )
                        self.logger.info("Model loaded from snapshot directory")
                    except Exception as snapshot_error:
                        # If snapshot loading also fails, try without local_files_only
                        # This allows it to download missing model config if needed
                        self.logger.warning(f"Failed to load from snapshot with local_files_only: {snapshot_error}")
                        self.logger.info("Trying to load model without local_files_only restriction...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            snapshot_dir,
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            low_cpu_mem_usage=True,  # Optimize memory usage with accelerate
                            use_cache=True,
                            local_files_only=False  # Allow downloading missing files
                        )
                        self.logger.info("Model loaded (may have downloaded missing files)")
                else:
                    # No snapshot directory, try loading with model name
                    self.logger.info(f"Loading model {self.model_name} from cache...")
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            low_cpu_mem_usage=True,  # Optimize memory usage with accelerate
                            use_cache=True,
                            local_files_only=True  # Use cache only
                        )
                        self.logger.info("Model loaded from cache")
                    except Exception as e:
                        # If that fails, try without local_files_only
                        self.logger.warning(f"Failed to load model with model name: {str(e)}")
                        self.logger.info("Trying to load model without local_files_only restriction...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            low_cpu_mem_usage=True,  # Optimize memory usage with accelerate
                            use_cache=True,
                            local_files_only=False  # Allow downloading
                        )
                        self.logger.info("Model loaded (may have downloaded missing files)")
            except (OSError, ValueError) as cache_error:
                # Cache load failed - try downloading if we have internet
                self.logger.warning(f"Failed to load from cache: {str(cache_error)}")
                self.logger.info("Attempting to download model (requires internet connection)...")
                try:
                    if self.tokenizer is None:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            local_files_only=False  # Allow downloading
                        )
                        self.logger.info("Tokenizer downloaded successfully")
                    
                    if self.model is None:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            low_cpu_mem_usage=True,  # Optimize memory usage with accelerate
                            use_cache=True,
                            local_files_only=False  # Allow downloading
                        )
                        self.logger.info("Model downloaded successfully")
                except ConnectionError as download_error:
                    # Download failed - this means we have no internet AND cache failed
                    # Try cache one more time in case it was a transient error
                    self.logger.warning(f"Download failed, retrying cache load: {str(download_error)}")
                    raise cache_error  # Re-raise the original cache error
            except ConnectionError as e:
                # Network/connection error - but model might be in cache
                # Try loading from cache with local_files_only=True
                self.logger.warning(f"Connection error, trying to load from cache: {str(e)}")
                try:
                    if self.tokenizer is None:
                        self.logger.info("Attempting to load tokenizer from cache...")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                    if self.model is None:
                        self.logger.info("Attempting to load model from cache...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            low_cpu_mem_usage=True,  # Optimize memory usage with accelerate
                            use_cache=True,
                            local_files_only=True
                        )
                    self.logger.info("Successfully loaded model from cache despite connection error")
                except Exception as cache_error:
                    # If cache loading also fails, provide helpful error message
                    error_msg = (
                        f"Failed to connect to HuggingFace and model not found in cache.\n\n"
                        f"Model cache location: {os.path.expanduser('~/.cache/huggingface/hub')}\n\n"
                        f"To download the model manually, run:\n"
                        f"  python -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; "
                        f"AutoTokenizer.from_pretrained('{self.model_name}', trust_remote_code=True); "
                        f"AutoModelForCausalLM.from_pretrained('{self.model_name}', trust_remote_code=True)\"\n\n"
                        f"Or use the download script:\n"
                        f"  python download_defog_model.py\n\n"
                        f"Connection error: {str(e)}\n"
                        f"Cache error: {str(cache_error)}"
                    )
                    raise ConnectionError(error_msg) from e
            except OSError as e:
                # Model not found or path issue
                error_str = str(e).lower()
                if "not the path to a directory" in error_str or "couldn't find it" in error_str or "no such file" in error_str:
                    # Before giving up, try loading from cache explicitly
                    self.logger.warning(f"OSError encountered, attempting to load from cache: {str(e)}")
                    try:
                        # Try loading from cache one more time
                        if self.tokenizer is None:
                            self.logger.info("Retrying tokenizer load from cache...")
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                self.model_name,
                                trust_remote_code=True,
                                local_files_only=True
                            )
                            self.logger.info("Tokenizer loaded from cache on retry")
                        
                        if self.model is None:
                            self.logger.info("Retrying model load from cache...")
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_name,
                                trust_remote_code=True,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                device_map="auto" if torch.cuda.is_available() else None,
                                low_cpu_mem_usage=True,  # Optimize memory usage with accelerate
                                use_cache=True,
                                local_files_only=True
                            )
                            self.logger.info("Model loaded from cache on retry")
                        
                        # If we got here, we successfully loaded from cache
                        self.logger.info("Successfully loaded model from cache after OSError")
                    except Exception as cache_retry_error:
                        # Cache retry also failed, provide error message
                        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                        hub_cache = os.path.join(cache_dir, "hub")
                        
                        # Log cache location for debugging
                        self.logger.warning(f"Model cache location: {hub_cache}")
                        self.logger.warning(f"Checking if model exists in cache...")
                        
                        # Try to find the model in cache
                        model_cache_path = os.path.join(hub_cache, f"models--{self.model_name.replace('/', '--')}")
                        if os.path.exists(model_cache_path):
                            self.logger.info(f"Found model cache directory: {model_cache_path}")
                            # Cache exists but loading failed - might be corrupted
                            error_msg = (
                                f"Model '{self.model_name}' found in cache but failed to load.\n\n"
                                f"Model cache location: {hub_cache}\n"
                                f"Cache directory exists: {model_cache_path}\n\n"
                                f"The cached model might be corrupted. Try re-downloading:\n"
                                f"  python download_defog_model.py\n\n"
                                f"Original error: {str(e)}\n"
                                f"Cache retry error: {str(cache_retry_error)}"
                            )
                        else:
                            self.logger.warning(f"Model cache directory not found: {model_cache_path}")
                            error_msg = (
                                f"Model '{self.model_name}' not found in cache.\n\n"
                                f"Model cache location: {hub_cache}\n\n"
                                f"To download the model, run:\n"
                                f"  python -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; "
                                f"AutoTokenizer.from_pretrained('{self.model_name}', trust_remote_code=True); "
                                f"AutoModelForCausalLM.from_pretrained('{self.model_name}', trust_remote_code=True)\"\n\n"
                                f"Or use the download script:\n"
                                f"  python download_defog_model.py\n\n"
                                f"Make sure you have internet connectivity and the model name is correct.\n"
                                f"Original error: {str(e)}\n"
                                f"Cache retry error: {str(cache_retry_error)}"
                            )
                        raise OSError(error_msg) from e
                else:
                    raise
            
            # Determine device for pipeline and ensure model is on GPU if available
            # Check if model was loaded with device_map (which handles device placement automatically via accelerate)
            model_has_device_map = hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None
            
            if torch.cuda.is_available():
                device = 0  # Use first GPU
                
                if not model_has_device_map:
                    # No device_map, explicitly move model to GPU
                    self.model = self.model.to(device)
                    self.logger.info(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
                else:
                    # Verify model is on GPU (device_map handles placement automatically)
                    first_param_device = next(self.model.parameters()).device
                    self.logger.info(f"Model loaded with device_map='auto' on device: {first_param_device}")
                    # When using device_map, don't set device in pipeline - it's handled automatically
                    device = None
                
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                device = -1  # Use CPU
                self.logger.info("Using CPU (GPU not available)")
            
            # Create pipeline
            # Note: When device_map is used, don't set device parameter - accelerate handles it
            pipeline_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "max_new_tokens": 300,
                "do_sample": False,
                "return_full_text": False,
                "num_beams": 5,
            }
            
            # Only set device if not using device_map (device_map handles device placement automatically)
            if device is not None:
                pipeline_kwargs["device"] = device
            
            self.pipe = pipeline("text-generation", **pipeline_kwargs)
            
            self.model_loaded = True
            device_str = "GPU" if torch.cuda.is_available() else "CPU"
            self.logger.info(f"Defog SQLCoder model loaded successfully on {device_str}")
        except (ConnectionError, OSError) as e:
            # Re-raise connection/OS errors with better messages
            raise
        except Exception as e:
            error_msg = (
                f"Failed to load Defog SQLCoder model '{self.model_name}': {str(e)}\n\n"
                f"Please ensure:\n"
                f"1. You have internet connectivity\n"
                f"2. The model name '{self.model_name}' is correct\n"
                f"3. You have sufficient disk space (~14GB for the model)\n"
                f"4. transformers and torch are properly installed\n\n"
                f"To download the model manually:\n"
                f"  python -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; "
                f"AutoTokenizer.from_pretrained('{self.model_name}'); "
                f"AutoModelForCausalLM.from_pretrained('{self.model_name}', trust_remote_code=True)\""
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _generate_prompt(self, question: str, schema: str) -> str:
        """Generate prompt from question and schema."""
        if self.prompt_file and self.prompt_file.exists():
            with open(self.prompt_file, "r", encoding='utf-8') as f:
                prompt_template = f.read()
            
            # Format the prompt
            prompt = prompt_template.format(
                user_question=question,
                table_metadata_string=schema
            )
        else:
            # Fallback to default prompt format
            prompt = f"""### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
"""
        return prompt
    
    def generate_sql(self, question: str, schema: str) -> Tuple[bool, str]:
        """
        Generate SQL query from natural language question using Defog SQLCoder.
        
        Args:
            question: Natural language question
            schema: Database schema in CREATE TABLE format
            
        Returns:
            Tuple of (success, sql_query or error_message)
        """
        try:
            # Load model if not already loaded
            if not self.model_loaded:
                self._load_model()
            
            # Generate prompt
            prompt = self._generate_prompt(question, schema)
            
            # Generate SQL query
            eos_token_id = self.tokenizer.eos_token_id
            generated_text = self.pipe(
                prompt,
                num_return_sequences=1,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
            )[0]["generated_text"]
            
            # Extract SQL query
            sql_query = (
                generated_text
                .split(";")[0]
                .split("```")[0]
                .split("[SQL]")[-1]
                .split("[/SQL]")[0]
                .strip()
            )
            
            # Clean up the query
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:].strip()
            if sql_query.startswith("```"):
                sql_query = sql_query[3:].strip()
            
            # Add semicolon if not present
            if sql_query and not sql_query.endswith(";"):
                sql_query += ";"
            
            if sql_query:
                self.logger.info(f"Generated SQL query: {sql_query[:100]}...")
                return True, sql_query
            else:
                return False, "Could not extract SQL query from response"
                
        except Exception as e:
            error_msg = f"Error generating SQL with Defog SQLCoder: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def check_availability(self) -> Tuple[bool, str]:
        """
        Check if Defog SQLCoder model is available.
        
        Returns:
            Tuple of (available, message)
        """
        try:
            # Try to load the model
            if not self.model_loaded:
                self.logger.info("Model not loaded, attempting to load...")
                self._load_model()
            
            if self.model_loaded:
                device = "GPU" if torch.cuda.is_available() else "CPU"
                return True, f"Defog SQLCoder ({self.model_name}) is available on {device}"
            else:
                return False, "Defog SQLCoder model failed to load"
        except (ConnectionError, OSError, RuntimeError) as e:
            # Return detailed error message for connection/model issues
            error_msg = str(e)
            self.logger.error(f"Error checking availability: {error_msg}", exc_info=True)
            return False, error_msg
        except Exception as e:
            error_msg = f"Defog SQLCoder not available: {str(e)}"
            self.logger.error(f"Unexpected error checking availability: {error_msg}", exc_info=True)
            return False, error_msg

