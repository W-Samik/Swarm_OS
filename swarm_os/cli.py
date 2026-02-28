import argparse
import sys
# Import from the new engine file
from swarm_os.engine import SwarmMaster, SwarmWorker

# MODEL PRESETS
MODEL_MAP = {
    "tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct"
}

def main():
    parser = argparse.ArgumentParser(description="Swarm-OS CLI")
    parser.add_argument('--role', choices=['A', 'B'], required=True)
    parser.add_argument('--model', default='tiny')
    args = parser.parse_args()
    
    model_id = MODEL_MAP.get(args.model, args.model)

    if args.role == 'B':
        worker = SwarmWorker(model_id=model_id)
        worker.start()
        
    elif args.role == 'A':
        master = SwarmMaster(model_id=model_id)
        try:
            master.connect()
            print("\nType 'exit' to quit.\n")
            while True:
                prompt = input("[User]: ")
                if prompt.lower() == 'exit': break
                
                print("[Swarm]: ", end="", flush=True)
                for chunk in master.generate(prompt):
                    print(chunk, end="", flush=True)
                print("\n")
        finally:
            master.close()

if __name__ == "__main__":
    main()