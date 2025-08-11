import modal

# Image that installs all dependencies using uv.
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "torch",
    "transformers", 
    "wandb", 
    "numpy",
    "datasets",
    "accelerate",
    "trl",
    "peft"
)

app = modal.App("gpt-training", image=image)

@app.function(gpu="H200:8", timeout=3600, secrets=[modal.Secret.from_name("wandb-secret")])
def train_gpt_grpo():
    # Import all dependencies inside the function to avoid conflicts
    import torch
    import wandb
    import numpy as np
    import json
    import re
    import os
    import sys
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    from trl import AutoModelForCausalLMWithValueHead, GRPOTrainer, GRPOConfig
    from dataclasses import dataclass
    from typing import List, Dict, Any, Optional, Tuple
    import time
    
    # Add regolf to path for imports
    sys.path.append("/regolf")
    
    # Import evaluation pipeline from mounted directory
    from regex.regex_evaluator import RegexEvaluator, create_evaluator
    from regex.reward_calculator import compute_grpo_rewards
    from prompts.agent_prompts.system_message import SYSTEM_MESSAGE
    from prompts.agent_prompts.user_message import USER_MESSAGE
    from prompts.agent_prompts.developer_message import DEVELOPER_MESSAGE
    
    # Initialize wandb
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.init(project="regolf-grpo-training", name="gpt-oss-20b-grpo")
    
    # Load gpt-oss-20b model and tokenizer
    model_name = "openai/gpt-oss-20b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with value head for GRPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load regex golf dataset
    print("Loading regex golf dataset...")
    with open("/regolf/data/validated_data/regex_golf_dataset_puzzle.json", "r") as f:
        regex_data = json.load(f)
    
    # Use subset for faster training
    regex_data = regex_data[:100]
    print(f"Using {len(regex_data)} regex golf problems")
    
    # Initialize regex evaluator with proper reward calculation
    evaluator = create_evaluator(
        length_penalty=0.01,
        correctness_mode="fractional",  # Use fractional for better learning signal
        invalid_reward=-10.0,
        strict_parsing=False,  # Allow flexible parsing
        verbose=False
    )
    
    # Prepare dataset for GRPO using proper prompts
    def format_regex_prompt(problem):
        """Format problem using the proper agent prompts"""
        # Format the user message with yes/no lists
        user_msg = USER_MESSAGE.format(
            yes_list=json.dumps(problem['yes']),
            no_list=json.dumps(problem['no'])
        )
        
        # Combine system, developer, and user messages
        full_prompt = SYSTEM_MESSAGE + DEVELOPER_MESSAGE + user_msg
        return full_prompt
    
    prompts = [format_regex_prompt(problem) for problem in regex_data]
    
    # Create dataset
    dataset = Dataset.from_dict({"query": prompts})
    
    # Reward function for GRPO using proper evaluation pipeline
    def compute_rewards(queries: List[str], responses: List[str], model_outputs) -> List[float]:
        """Compute rewards for generated regex patterns using proper evaluation pipeline"""
        rewards = []
        solutions = []
        
        for i, (query, response) in enumerate(zip(queries, responses)):
            problem = regex_data[i]  # Get corresponding problem
            
            # Use the proper regex evaluator to evaluate the model output
            solution = evaluator.evaluate_single(
                model_output=response,
                yes_strings=problem["yes"],
                no_strings=problem["no"]
            )
            
            solutions.append(solution)
            rewards.append(solution.reward)
            
            # Log detailed metrics for first few examples
            if i < 5:
                solution_dict = solution.to_dict()
                log_data = {f"example_{i}_{k}": v for k, v in solution_dict.items()}
                wandb.log(log_data)
        
        # Log parsing/validation success rates
        valid_solutions = [s for s in solutions if s.is_valid()]
        parse_errors = [s for s in solutions if s.parse_error is not None]
        validation_errors = [s for s in solutions if s.validation_error is not None]
        
        wandb.log({
            "valid_solutions_rate": len(valid_solutions) / len(solutions),
            "parse_error_rate": len(parse_errors) / len(solutions),
            "validation_error_rate": len(validation_errors) / len(solutions)
        })
        
        # Normalize rewards for stable training
        rewards = compute_grpo_rewards(rewards, normalize=True)
        
        # Log reward statistics
        wandb.log({
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards)
        })
        
        return rewards
    
    # GRPO training configuration
    grpo_config = GRPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        ppo_epochs=4,
        max_grad_norm=1.0,
        target_kl=0.1,
        forward_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        warmup_steps=100,
        report_to="wandb"
    )
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_fn=compute_rewards,
    )
    
    print("Starting GRPO training...")
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model("/regolf/models/gpt-oss-20b-grpo-final")
    
    # Test on a few examples
    print("\n=== Testing trained model ===")
    test_problems = regex_data[:3]
    
    for i, problem in enumerate(test_problems):
        prompt = format_regex_prompt(problem)
        
        # Generate response
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.pretrained_model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_new_tokens=50, 
                temperature=0.7, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        # Evaluate response using proper evaluation pipeline
        solution = evaluator.evaluate_single(
            model_output=response,
            yes_strings=problem["yes"],
            no_strings=problem["no"]
        )
        
        print(f"\nTest {i+1}:")
        print(f"  Generated output: {response[:100]}...")
        if solution.is_valid():
            print(f"  Compiled pattern: {solution.pattern_str}")
            print(f"  Accuracy: {solution.eval_result.accuracy:.3f}")
            print(f"  Reward: {solution.reward:.3f}")
            print(f"  Is perfect: {solution.eval_result.is_perfect()}")
        else:
            print(f"  Invalid solution!")
            print(f"  Parse error: {solution.parse_error}")
            print(f"  Validation error: {solution.validation_error}")
            print(f"  Reward: {solution.reward:.3f}")
    
    wandb.finish()
    return "GRPO training completed successfully!"

@app.local_entrypoint()
def main():
    result = train_gpt_grpo.remote()  # runs in the cloud
    print(result)
